# main.py
import os
import sys

def main():
    # PyInstaller onefile 解包目录；非打包时等于当前目录
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app.py")

    # 确保运行目录与 import 路径正确
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    os.chdir(base_dir)

    # 关闭 Streamlit 开发模式，避免 server.port 冲突
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    # 用环境变量设置端口（不要用 --server.port）
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"

    from streamlit.web import cli as stcli

    # 启动 Streamlit（不传 server.port）
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless=true",
        "--browser.serverAddress=localhost",
    ]

    sys.exit(stcli.main())

if __name__ == "__main__":
    main()