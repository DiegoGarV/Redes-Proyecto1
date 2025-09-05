# cliente/cli.py — MVP: conexión a Claude, memoria de sesión y logging JSONL
# Ejecuta:  python -m cliente.cli chat
# Requisitos (añade a requirements.txt e instala): anthropic, typer[all], rich, python-dotenv, loguru

from __future__ import annotations
import os, json, uuid, datetime as dt
from pathlib import Path
from typing import List, Dict, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from loguru import logger

# --- LLM (Claude, Anthropic) ---
try:
    import anthropic  # Anthropic SDK oficial
except Exception as e:  # pragma: no cover
    raise SystemExit("Falta 'anthropic'. Agrega 'anthropic' a requirements.txt y pip install -r requirements.txt")

app = typer.Typer(add_completion=False)
console = Console()

# --- Carga .env ---
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")  # ver docs modelos

# --- Rutas ---
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "host.jsonl"

# Config loguru: JSON por línea
logger.remove()
logger.add(LOG_FILE, serialize=True, backtrace=False, diagnose=False, rotation="5 MB", retention=5)

# -------------------------------------------------------
# Sesión de chat con memoria (lista de turnos)
# -------------------------------------------------------
class Session:
    def __init__(self, sid: str | None = None, system_prompt: str | None = None):
        self.id = sid or str(uuid.uuid4())
        self.system = system_prompt or (
            """
            Eres "MCP Modpack Assistant", un asistente por terminal enfocado en ayudar a construir
            y mantener modpacks de Minecraft (validación de versiones/loader, dependencias, manifests).
            Habla en español claro y conciso. Mantén el contexto entre turnos.
            Si el usuario hace preguntas generales (e.g., historia/biografías), respóndelas brevemente.
            """
        ).strip()
        self.turns: List[Dict[str, str]] = []  # cada item: {role: 'user'|'assistant', content: str}

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        # Mantén últimas 20 interacciones (40 mensajes aprox.)
        if len(self.turns) > 40:
            self.turns = self.turns[-40:]

    def messages(self) -> List[Dict[str, Any]]:
        # La API de Messages usa lista de turnos user/assistant y un system aparte.
        return self.turns.copy()

# -------------------------------------------------------
# Cliente Claude (Anthropic Messages API)
# -------------------------------------------------------
class ClaudeClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = api_key or ANTHROPIC_API_KEY
        if not key:
            raise RuntimeError("Falta ANTHROPIC_API_KEY en .env")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model or ANTHROPIC_MODEL

    def complete(self, session: Session, max_tokens: int = 600) -> Dict[str, Any]:
        """Envía la conversación a Claude y devuelve texto + metadatos."""
        # Log request (LLM)
        logger.bind(peer="llm", direction="send", session=session.id).info({
            "event": "messages.create",
            "model": self.model,
            "system_len": len(session.system or ""),
            "turns": len(session.turns),
        })
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=session.system,
            messages=session.messages(),
        )
        # Extrae texto concatenando bloques tipo 'text'
        text = "".join([b.text for b in resp.content if getattr(b, "type", None) == "text"]) or ""
        # Log response (LLM)
        logger.bind(peer="llm", direction="recv", session=session.id).info({
            "event": "message",
            "id": resp.id,
            "stop_reason": resp.stop_reason,
            "usage": getattr(resp, "usage", None) and resp.usage.__dict__,
        })
        return {"id": resp.id, "text": text, "usage": getattr(resp, "usage", None)}

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
@app.command()
def chat():
    """Inicia un chat con memoria de sesión y logging a logs/host.jsonl"""
    session = Session()
    llm = ClaudeClient()
    console.print(f"[bold]MCP Modpack Assistant[/bold] — sesión: [cyan]{session.id}[/cyan]\n(Escribe 'salir' para terminar)\n")

    while True:
        try:
            user = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Adiós.[/dim]")
            break
        if not user:
            continue
        if user.lower() in {"salir", "exit", ":q"}:
            console.print("[dim]Adiós.[/dim]")
            break

        # Log entrada del usuario
        logger.bind(peer="user", direction="send", session=session.id).info({"event": "input", "text": user})
        session.add("user", user)

        try:
            out = llm.complete(session)
        except Exception as e:  # captura errores de red/API
            logger.bind(peer="llm", direction="recv", session=session.id).error({"event": "error", "msg": str(e)})
            console.print(f"[red]Error de LLM:[/red] {e}")
            continue

        session.add("assistant", out["text"])  # memoria de la respuesta
        console.print(Markdown(out["text"]))

@app.command()
def logs(tail: int = typer.Option(20, help="Cuántas líneas mostrar")):
    """Muestra las últimas N líneas del log JSONL (incluye LLM y, luego, MCP)."""
    if not LOG_FILE.exists():
        console.print("[yellow]No hay log aún.[/yellow]")
        raise typer.Exit(code=0)
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    for ln in lines[-tail:]:
        try:
            obj = json.loads(ln)
            ts = obj.get("time", dt.datetime.now().isoformat())
            peer = obj.get("extra", {}).get("peer", "?")
            direction = obj.get("extra", {}).get("direction", "?")
            payload = obj.get("message", {})
            console.print(f"[dim]{ts}[/dim] [cyan]{peer}[/cyan] [{direction}] {json.dumps(payload, ensure_ascii=False)}")
        except Exception:
            console.print(ln)

if __name__ == "__main__":
    app()
