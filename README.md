# Servidores MCP
## Generalidades
Despues de clonar el repo
Ir a la carpeta main del proyecto
``` 
cd modpack
```

Entrar en el entorno virtual
``` 
.venv\Scripts\activate
```

Descargar uv
```
pip install uv
```

## Servidor local para modpacks de Minecraft personalizado
Instalar el resto de requisitos
```
uv pip install -r requirements.txt
```

Para que el MCP funcione se debe crear un archivo *.env* con 2 llaves
```
CURSEFORGE_API_KEY= *COLOCAR AQUÍ LA LLAVE PARA LA API DE CURSEFORGE*
ANTHROPIC_API_KEY= *COLOCAR AQUÍ LA LLAVE PARA LA API DE ANTHROPIC/CLAUDE*
```

Asegurar que el servidor funcione
```
uv run mcp dev server/modpack_server.py
```
Se debe permitir las instalaciones requeridas. Esto solo asegura que el servidor MCP funciona sin problemas. La ventana automática que se abre se puede cerrar y el servidor se puede apagar. El cliente luego se encargará de levantarlo.

Correr al cliente local
```
uv run client/client.py 
```
**NOTA:** si lo primero que se hizo fue esto, se mostrará un mensaje de que no se pudo conectar al servidor remoto. Esto no es necesario para que el chatbot funcione correctamente. Abajo se explica como iniciar ese servidor.

## Servidor remoto
