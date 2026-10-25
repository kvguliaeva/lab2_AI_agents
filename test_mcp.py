
import asyncio
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    print("BRIDALFLOW AI — ПРОВЕРКА MCP")

    # Папка с учебными правилами студии
    folder = (
        Path(__file__).resolve().parent / "studio_docs"
    )

    if not (folder / "studio_rules.txt").is_file():
        raise FileNotFoundError(
            "Не найден studio_docs/studio_rules.txt"
        )

    # Подключение готового MCP-сервера
    client = MultiServerMCPClient({
        "filesystem": {
            "transport": "stdio",
            "command": "docker",
            "args": [
                "run",
                "--rm",
                "-i",
                "--mount",
                (
                    f"type=bind,src={folder},"
                    "dst=/projects/studio_docs,readonly"
                ),
                "mcp/filesystem",
                "/projects/studio_docs"
            ]
        }
    })

    print("\nПодключаем MCP-сервер...")

    # Получаем инструменты сервера
    tools = await client.get_tools()

    print("\nДоступные инструменты:")

    for tool in tools:
        print("-", tool.name)

    # Находим инструмент чтения файла
    read_tool = next(
        (
            tool for tool in tools
            if tool.name == "read_file"
        ),
        None
    )

    if read_tool is None:
        raise RuntimeError(
            "MCP-инструмент read_text_file не найден"
        )

    print("\nЧитаем правила студии через MCP...")

    # Реальный вызов MCP-инструмента
    result = await read_tool.ainvoke({
        "path": "/projects/studio_docs/studio_rules.txt"
    })

    print("\nСОДЕРЖИМОЕ ФАЙЛА:")
    print(result)

    print("\nMCP-инструмент успешно вызван!")


if __name__ == "__main__":
    asyncio.run(main())
