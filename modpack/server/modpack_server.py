"""
server/modpack_server.py — MCP server (STDIO) para asistir con modpacks de Minecraft
"""
from __future__ import annotations
import os
import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ----------------- Config & logging -----------------
load_dotenv()

# Log a stderr y a archivo (no stdout)
logger = logging.getLogger("modpack_server")
logger.setLevel(logging.INFO)
_stderr_h = logging.StreamHandler()  # seguro en STDIO servers
_stderr_h.setLevel(logging.INFO)
logger.addHandler(_stderr_h)

# Opcional: log a archivo
try:
    os.makedirs("logs", exist_ok=True)
    _file_h = logging.FileHandler("logs/modpack_server.log", encoding="utf-8")
    _file_h.setLevel(logging.INFO)
    logger.addHandler(_file_h)
except Exception:
    pass

# ----------------- CurseForge client (async) -----------------
CF_BASE = os.getenv("CURSEFORGE_BASE", "https://api.curseforge.com")
CF_KEY = os.getenv("CURSEFORGE_API_KEY")
MINECRAFT_GAME_ID = int(os.getenv("CURSEFORGE_MC_ID", "432"))  # 432 = Minecraft
USER_AGENT = os.getenv("CF_USER_AGENT", "mcp-modpack-assistant/0.1 (academic)")

if not CF_KEY:
    raise RuntimeError("Falta CURSEFORGE_API_KEY en .env")

# ----------------- MCP server -----------------
mcp = FastMCP("modpack")

async def _cf_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{CF_BASE}/v1{path}"
    headers = {"Accept": "application/json", "x-api-key": CF_KEY, "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url, params=params or {}, headers=headers)
        r.raise_for_status()
        return r.json()

# ----------------- Tools -----------------
@mcp.tool()
async def mc_versions(limit: int = 10) -> List[str]:
    """Devuelve las últimas versiones de Minecraft conocidas por CurseForge (ej. '1.20.1')."""
    logger.info("mc_versions(limit=%s)", limit)
    data = await _cf_get("/minecraft/modloader")
    versions = []
    for v in data.get("data", []):
        gv = v.get("gameVersion")
        if gv and gv not in versions:
            versions.append(gv)
    versions = list(reversed(versions))
    return versions[:limit]

@mcp.tool()
async def search_mods(query: str, page_size: int = 5, mc_version: str = None) -> str:
    """
    Busca mods en CurseForge por nombre. Filtra solo mods (no modpacks).
    """
    params = {
        "gameId": 432,        # Minecraft
        "classId": 6,         # Solo mods
        "searchFilter": query,
        "sortField": 2,       # Popularidad
        "sortOrder": "desc",
        "pageSize": page_size,
    }
    if mc_version:
        params["gameVersion"] = mc_version

    data = await _cf_get("/mods/search", params=params)

    if not data.get("data"):
        return f"No se encontraron mods para '{query}'."

    results = [
        f"[{m['id']}] {m['name']} ({m['slug']})"
        for m in data["data"]
    ]
    return "\n".join(results)


@mcp.tool()
async def mod_files(mod_id: int, page_size: int = 10, index: int = 0) -> str:
    """Lista archivos de un mod (muestra fileId, nombre y gameVersions)."""
    logger.info("mod_files(mod_id=%s, page_size=%s, index=%s)", mod_id, page_size, index)
    params = {"pageSize": max(1, min(page_size, 50)), "index": max(0, index)}
    data = await _cf_get(f"/mods/{mod_id}/files", params=params)
    items = data.get("data", [])
    if not items:
        return "No hay archivos para este mod o paginación vacía."
    lines = [f"Archivos de mod {mod_id}:"]
    for f in items:
        fid = f.get("id")
        disp = f.get("displayName")
        gvs = ", ".join(f.get("gameVersions", [])[:6])
        lines.append(f"- fileId={fid} — {disp} — versions: {gvs}")
    return "\n".join(lines)

@mcp.tool()
async def make_manifest(minecraft: str, loader: str, files_json: str) -> str:
    """Genera un manifest.json básico para CurseForge.

    Args:
      minecraft: versión, ej. '1.20.1'
      loader: 'forge' | 'fabric' | 'quilt' | 'neoforge'
      files_json: JSON con lista de objetos {"projectID": int, "fileID": int}
    """
    logger.info("make_manifest(mc=%s, loader=%s)", minecraft, loader)
    try:
        files: List[Dict[str, int]] = json.loads(files_json)
    except Exception:
        return "files_json inválido; debe ser JSON con [{projectID,fileID}]."

    manifest = {
        "minecraft": {
            "version": minecraft,
            "modLoaders": [{"id": f"{loader}", "primary": True}],
        },
        "files": [
            {"projectID": int(x["projectID"]), "fileID": int(x["fileID"]), "required": True}
            for x in files
        ],
        "manifestType": "minecraftModpack",
        "manifestVersion": 1,
        "name": f"Modpack {minecraft}-{loader}",
        "version": "0.1.0",
        "author": "diego",
        "overrides": "overrides"
    }

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return f"Manifest generado en {out_path}.\n\nContenido:\n" + json.dumps(manifest, indent=2, ensure_ascii=False)

# ----------------- Main -----------------
if __name__ == "__main__":
    logger.info("Iniciando MCP server 'modpack' (STDIO)")
    mcp.run(transport="stdio")
