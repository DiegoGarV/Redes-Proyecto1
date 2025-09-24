from __future__ import annotations
import os
import sys
import re
import json
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.panel import Panel
from rich import box

# Anthropic SDK
import anthropic

# MCP Python client (incluido con mcp[cli])
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import shutil

# ---------------------- Config & logging ----------------------
load_dotenv()

console = Console()
LOG_DIR = os.path.join("logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("chatbot_cli")
logger.setLevel(logging.INFO)
_hdlr = logging.FileHandler(os.path.join(LOG_DIR, "chatbot_cli.log"), encoding="utf-8")
_hdlr.setLevel(logging.INFO)
logger.addHandler(_hdlr)

# ---------------------- LLM Config ----------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    console.print("[bold red]Falta ANTHROPIC_API_KEY en .env[/bold red]")
    sys.exit(1)

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ---------------------- MCP Server Config ----------------------
# Permite forzar el ejecutable y el script por variables de entorno
MCP_CMD = os.getenv("MCP_CMD")  # p.ej. C:\Users\...\modpack\.venv\Scripts\python.exe
MCP_ARGS = os.getenv("MCP_ARGS")  # p.ej. C:\Users\...\modpack\server\modpack_server.py

# Si no se definió, tratamos de deducir rutas relativas al repo
if not MCP_CMD:
    # Asumimos el Python del venv actual
    MCP_CMD = sys.executable
if not MCP_ARGS:
    MCP_ARGS = os.path.join(os.path.dirname(__file__), "..", "server", "modpack_server.py")
    MCP_ARGS = os.path.abspath(MCP_ARGS)

def _anthropic_tool_from_mcp(mcp_tool) -> dict:
    return {
        "name": mcp_tool.name,
        "description": (mcp_tool.description or "")[:500],
        "input_schema": mcp_tool.inputSchema or {"type": "object"},
    }

async def build_anthropic_tools(session) -> list[dict]:
    listed = await session.list_tools()
    tools = []
    for t in (listed.tools or []):
        tools.append(_anthropic_tool_from_mcp(t))
    return tools

# ---------------------- Utilidades ----------------------
@dataclass
class MCPCallResult:
    tool: str
    arguments: Dict[str, Any]
    ok: bool
    response: str

@dataclass
class ToolBinding:
    session: ClientSession
    real_name: str

def _sanitize_name(s: str) -> str:
    # Cumple el regex de Anthropic
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)[:128]

async def connect_mcp(exit_stack: AsyncExitStack) -> ClientSession:
    """Establece conexión STDIO con el servidor MCP local"""
    params = StdioServerParameters(command=MCP_CMD, args=[MCP_ARGS])

    # stdio_client devuelve (reader, writer)
    reader, writer = await exit_stack.enter_async_context(stdio_client(params))

    session = await exit_stack.enter_async_context(ClientSession(reader, writer))
    await session.initialize()
    return session

async def _spawn_with_candidates(exit_stack: AsyncExitStack, candidates: list[tuple[str, list[str]]]) -> ClientSession:
    last_err = None
    for cmd, args in candidates:
        try:
            params = StdioServerParameters(command=cmd, args=args)
            r, w = await exit_stack.enter_async_context(stdio_client(params))
            s = await exit_stack.enter_async_context(ClientSession(r, w))
            await s.initialize()
            return s
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "No pude iniciar el servidor MCP externo.\n"
        "Prueba manualmente en tu terminal:\n"
        "  npx -y @modelcontextprotocol/server-filesystem --help\n"
        "  npx -y @modelcontextprotocol/server-github --help\n\n"
        f"Último error: {last_err}"
    )

async def connect_fs(exit_stack: AsyncExitStack) -> ClientSession:
    roots = os.environ.get("FS_ROOT") or os.getcwd()
    perms = os.environ.get("FS_MODE", "readwrite")  # read|write|readwrite

    # Algunas implementaciones tratan arg posicional como raíz; otras aceptan flags.
    flag_sets = [
        # 1) SOLO POSICIONAL (más compatible): <root> [--permissions X]
        ([roots, "--permissions", perms]),
        ([roots, "--mode", perms]),
        ([roots]),

        # 2) Con flags (por si tu build sí los soporta)
        (["--roots", roots, "--permissions", perms]),
        (["--root", roots, "--permissions", perms]),
        (["--roots", roots, "--mode", perms]),
        (["--root", roots, "--mode", perms]),
    ]

    candidates: list[tuple[str, list[str]]] = []
    for flags in flag_sets:
        # 1) npx (el más robusto en Windows)
        candidates.append(("npx", ["-y", "@modelcontextprotocol/server-filesystem", *flags]))
        # 2) bin en PATH (si lo instalaste con -g)
        candidates.append(("mcp-server-filesystem", flags))

    # 3) wrappers .cmd/.ps1 típicos en Windows
    fs_cmd = shutil.which("mcp-server-filesystem.cmd") or shutil.which("mcp-server-filesystem.ps1")
    if fs_cmd:
        for flags in flag_sets:
            candidates.append((fs_cmd, flags))

    return await _spawn_with_candidates(exit_stack, candidates)

async def connect_github(exit_stack: AsyncExitStack) -> ClientSession:
    args = []
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        args += ["--token", token]

    candidates = [
        ("npx", ["-y", "@modelcontextprotocol/server-github", *args]),
        ("mcp-server-github", args),
    ]
    gh_cmd = shutil.which("mcp-server-github.cmd") or shutil.which("mcp-server-github.ps1")
    if gh_cmd:
        candidates.append((gh_cmd, args))

    return await _spawn_with_candidates(exit_stack, candidates)

async def list_all_tools(sessions: dict[str, ClientSession]) -> list[str]:
    names: list[str] = []
    for prefix, sess in sessions.items():
        lt = await sess.list_tools()
        for t in (lt.tools or []):
            names.append(f"{prefix}.{t.name}")  # p.ej., fs.read_file, github.search, modpack.mc_versions
    return names

async def build_tool_registry(sessions: dict[str, ClientSession]):
    tools_for_llm = []
    dispatch: dict[str, ToolBinding] = {}

    for prefix, sess in sessions.items():
        listed = await sess.list_tools()
        for t in (listed.tools or []):
            safe_prefix = _sanitize_name(prefix)
            safe_tool   = _sanitize_name(t.name)
            exposed     = f"{safe_prefix}__{safe_tool}"

            tools_for_llm.append({
                "name": exposed,
                "description": (t.description or "")[:500],
                "input_schema": t.inputSchema or {"type": "object"},
            })
            dispatch[exposed] = ToolBinding(session=sess, real_name=t.name)

    return tools_for_llm, dispatch

async def list_tools(session: ClientSession) -> List[str]:
    lt = await session.list_tools()
    return [t.name for t in lt.tools or []]

async def call_tool(session: ClientSession, name: str, arguments: Dict[str, Any]) -> MCPCallResult:
    """Llama una tool MCP y devuelve el resultado como texto plano."""
    logger.info("[MCP] call_tool %s(%s)", name, json.dumps(arguments, ensure_ascii=False))
    res = await session.call_tool(name, arguments=arguments)
    # El contenido suele venir como una lista de bloques "text"
    text_chunks: List[str] = []
    for item in (res.content or []):
        if getattr(item, "type", None) == "text":
            text_chunks.append(item.text)
        elif isinstance(item, dict) and item.get("type") == "text":
            text_chunks.append(item.get("text", ""))
    out = "\n".join(text_chunks).strip()
    logger.info("[MCP] response %s: %s", name, out[:4000])
    return MCPCallResult(tool=name, arguments=arguments, ok=True, response=out)

async def run_llm_with_tools(history: list[dict], sessions: dict[str, ClientSession]) -> Tuple[str, list[dict]]:
    tools_for_llm, dispatch = await build_tool_registry(sessions)

    msg = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": m["role"], "content": m["content"]} for m in history if m["role"] != "system"],
        tools=tools_for_llm,
        tool_choice={"type": "auto"},
    )

    added_messages = []
    content_blocks = msg.content or []
    tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

    while tool_uses:
        tool_results_blocks = []
        for tu in tool_uses:
            exposed_name = tu.name  # ej: "fs__read_file"
            tb = dispatch.get(exposed_name)
            if not tb:
                out_text = f"(tool desconocida: {exposed_name})"
            else:
                try:
                    mcp_res = await tb.session.call_tool(tb.real_name, arguments=(tu.input or {}))
                    out_texts = []
                    for it in (mcp_res.content or []):
                        if getattr(it, "type", None) == "text":
                            out_texts.append(it.text)
                        elif isinstance(it, dict) and it.get("type") == "text":
                            out_texts.append(it.get("text", ""))
                    out_text = "\n".join(out_texts).strip() or "(sin contenido)"
                except Exception as e:
                    out_text = f"(error al ejecutar tool {exposed_name}: {e})"

            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": out_text[:100000],
            })

        history.append({"role": "assistant", "content": content_blocks})
        history.append({"role": "user", "content": tool_results_blocks})
        added_messages.extend([
            {"role": "assistant", "content": content_blocks},
            {"role": "user", "content": tool_results_blocks},
        ])

        msg = anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": m["role"], "content": m["content"]} for m in history if m["role"] != "system"],
            tools=tools_for_llm,
            tool_choice={"type": "auto"},
        )
        content_blocks = msg.content or []
        tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

    final_text = "".join(getattr(b, "text", "") for b in content_blocks if getattr(b, "type", None) == "text").strip()
    return final_text, added_messages

# ---------------------- UX helpers ----------------------
def print_header():
    console.print(
        Panel.fit(
            "Chatbot para buscar mods de Minecraft y crear un modpack propio.",
            title="Host + MCP",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

def pretty_block(title: str, body: str):
    console.print(Panel.fit(body if body.strip() else "(sin contenido)",
                            title=title, border_style="green", box=box.ROUNDED))

def table_tools(tools: List[str]):
    t = Table(title="Tools MCP disponibles", box=box.SIMPLE_HEAVY)
    t.add_column("#", justify="right")
    t.add_column("tool")
    for i, name in enumerate(tools, 1):
        t.add_row(str(i), name)
    console.print(t)

# ---------------------- Chat loop ----------------------
SYSTEM_PROMPT = (
    "Eres un asistente que ayuda a encontrar mods de Minecraft y a generar un manifest "
    "para crear un modpack propio. Habla de forma natural. "
    "NO menciones nombres de comandos ni herramientas internas (p. ej., mc_versions, "
    "search_mods, mod_files, make_manifest). Si necesitas información técnica, "
    "pide detalles en lenguaje natural (por ejemplo: '¿Qué versión de Minecraft quieres usar?' "
    "o '¿Quieres optimización de rendimiento o mapas?'). "
    "Cuando el usuario pida crear un manifest, confirma versión de Minecraft y tipo de loader "
    "(Forge/Fabric/Quilt) y procede. No inventes IDs."
    "Mantén tus respuestas concisas y enfocadas pero que también se sienta como una guía."
)

async def chat_loop():
    print_header()

    # Creamos un stack para manejar los context managers async (stdio_client y ClientSession)
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()
    
    try:
        modpack = await connect_mcp(exit_stack)   # server local de CurseForge
        fs = await connect_fs(exit_stack)         # filesystem
        github = await connect_github(exit_stack) # GitHub

        # Conectar al servidor MCP usando el exit_stack
        sessions = {"modpack": modpack, "fs": fs, "github": github}
        console.print("[green]Conectado a MCP (modpack, fs, github)[/green]\n")

        history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        while True:
            user_in = Prompt.ask("[bold yellow]Tú[/bold yellow]").strip()
            if not user_in:
                continue
            if user_in.lower() in {"exit", "quit", ":q"}:
                break

            # ---------------------- Comandos MCP ----------------------
            if user_in.startswith("::"):
                parts: List[str] = user_in[2:].strip().split()
                if not parts:
                    continue
                cmd = parts[0]
                args = parts[1:]

                if cmd == "tools":
                    names = await list_all_tools(sessions)
                    table_tools(names)
                    continue

                if cmd == "mc_versions":
                    limit = int(args[0]) if args else 10
                    res = await call_tool(modpack, "mc_versions", {"limit": limit})
                    pretty_block("mc_versions", res.response)
                    continue

                if cmd == "search_mods":
                    if not args:
                        console.print("[red]Uso:[/red] ::search_mods <query> [page_size] [mc_version]")
                        continue
                    query = args[0]
                    page_size = int(args[1]) if len(args) >= 2 else 5
                    mc_version = args[2] if len(args) >= 3 else None
                    payload = {"query": query, "page_size": page_size}
                    if mc_version:
                        payload["mc_version"] = mc_version
                    res = await call_tool(modpack, "search_mods", payload)
                    pretty_block("search_mods", res.response)
                    continue

                if cmd == "mod_files":
                    if not args:
                        console.print("[red]Uso:[/red] ::mod_files <mod_id> [page_size]")
                        continue
                    mod_id = int(args[0])
                    page_size = int(args[1]) if len(args) >= 2 else 10
                    res = await call_tool(modpack, "mod_files", {"mod_id": mod_id, "page_size": page_size})
                    pretty_block("mod_files", res.response)
                    continue

                if cmd == "make_manifest":
                    if len(args) < 3:
                        console.print("[red]Uso:[/red] ::make_manifest <mc_version> <loader> <files_json>")
                        console.print("Ejemplo: ::make_manifest 1.20.1 forge "
                                      "'[{\"projectID\":238222,\"fileID\":4632296}]'")
                        continue
                    mc_version = args[0]
                    loader = args[1]
                    files_json = " ".join(args[2:])  # por si viene con espacios
                    res = await call_tool(modpack, "make_manifest",
                                          {"minecraft": mc_version, "loader": loader, "files_json": files_json})
                    pretty_block("make_manifest", res.response)
                    continue

                if cmd == "fstools":
                    names = await list_tools(fs)
                    table_tools([f"fs.{n}" for n in names])
                    continue

                if cmd == "ghtools":
                    names = await list_tools(github)
                    table_tools([f"github.{n}" for n in names])
                    continue

                if cmd == "modtools":
                    names = await list_tools(modpack)
                    table_tools([f"modpack.{n}" for n in names])
                    continue

                console.print(f"[red]Comando no reconocido:[/red] {cmd}")
                continue

            # ---------------------- Mensaje normal a LLM ----------------------
            history.append({"role": "user", "content": user_in})
            logger.info("[LLM] user: %s", user_in)

            text_out, extra_msgs = await run_llm_with_tools(history, sessions)
            history.extend(extra_msgs)  # guarda la traza tool_use/tool_result en el historial

            text_out = text_out.strip()

            history.append({"role": "assistant", "content": text_out})
            pretty_block("Claude", text_out)
            logger.info("[LLM] assistant: %s", text_out[:4000])

    finally:
        # Cierra ordenadamente la sesión y el stdio transport
        await exit_stack.aclose()
        console.print("\n[cyan]Conexión MCP cerrada. ¡Hasta luego![/cyan]")


if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print()
