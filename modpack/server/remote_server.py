# modpack/server/remote_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("remote-trivial")

@mcp.tool()
async def ping(text: str) -> str:
    """Echo simple para probar el remoto."""
    return f"pong: {text}"

@mcp.tool()
async def manifest_summary(mc_version: str, loader: str) -> str:
    """Resumen de manifest (trivial) para tu tema de modpacks."""
    return f"Manifest trivial → Minecraft {mc_version}, loader {loader}"

if __name__ == "__main__":
    # Local opcional (no se usa en Render); stdio en local.
    mcp.run()