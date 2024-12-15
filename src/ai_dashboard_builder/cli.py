"""Command-line interface for the AI Dashboard Builder."""

import argparse

from ai_dashboard_builder.app import app

DEFAULT_PORT = 8050
DEFAULT_HOST = "localhost"


def prod_server(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST):
    """Run the AI Dashboard Builder."""
    app.run(
        debug=False,
        host=host,
        port=port,
        dev_tools_hot_reload=False,
        dev_tools_ui=False,
        dev_tools_props_check=False,
    )


def dev_server(port: int = DEFAULT_PORT, host: str = DEFAULT_HOST):
    """Run the AI Dashboard Builder in development mode."""
    app.run(debug=True, host=host, port=port)


def main():
    """Run the AI Dashboard Builder."""
    parser = argparse.ArgumentParser(description="Run the AI Dashboard Builder.")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run the AI Dashboard Builder in development mode.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port number for the AI Dashboard Builder.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host address for the AI Dashboard Builder.",
    )
    args = parser.parse_args()

    if args.dev:
        dev_server(args.port, args.host)
    else:
        prod_server(args.port, args.host)


# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    main()
