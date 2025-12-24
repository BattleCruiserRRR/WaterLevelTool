# main.py
import os
import sys

def main():
    # 让 PyInstaller 打包后也能正确找到 app.py
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")

    # 运行 streamlit
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--server.port=8501",
        "--browser.serverAddress=localhost",
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()