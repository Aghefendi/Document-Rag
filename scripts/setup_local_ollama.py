import subprocess
import sys
import time

TEXT_MODELS = [
    "phi3:mini",
    "qwen2.5:3b",
    "llama3.2",
]

EMBED_MODELS = [
    "nomic-embed-text",
]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def container_status() -> str | None:
    info = run(["docker", "inspect", "-f", "{{.State.Status}}", "ollama"])
    if info.returncode != 0:
        return None
    return info.stdout.strip()


def ensure_container_running() -> bool:
    # Is docker up?
    probe = run(["docker", "ps"])
    if probe.returncode != 0:
        print("Docker is not available. Please start Docker Desktop first.")
        print(probe.stdout)
        return False

    status = container_status()
    if status == "running":
        return True
    if status in {"exited", "created", "paused"}:
        print(f"Starting existing Ollama container (status={status})...")
        start = run(["docker", "start", "ollama"])
        if start.returncode != 0:
            print(start.stdout)
            return False
        return True

    # Not present -> create
    print("Creating new Ollama container...")
    create = run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "ollama",
            "-p",
            "11434:11434",
            "-v",
            "ollama:/root/.ollama",
            "ollama/ollama",
        ]
    )
    if create.returncode != 0:
        print(create.stdout)
        return False

    # wait a moment for daemon to start
    time.sleep(3)
    status = container_status()
    if status != "running":
        print(f"Container did not start (status={status}). Logs:")
        logs = run(["docker", "logs", "--tail", "50", "ollama"])
        print(logs.stdout)
        return False

    return True


def pull_first_available(models: list[str], label: str) -> str:
    for model in models:
        print(f"Pulling {label} model candidate '{model}' ...")
        res = run(["docker", "exec", "-i", "ollama", "ollama", "pull", model])
        print(res.stdout)
        if res.returncode == 0:
            print(f"Using {label} model: {model}")
            return model
        else:
            print(f"{label} model '{model}' failed, trying next...")
    print(f"No {label} model could be pulled. Please check names or network.")
    sys.exit(1)


def main() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    ok = ensure_container_running()
    if not ok:
        sys.exit(1)
    text_model = pull_first_available(TEXT_MODELS, "text")
    embed_model = pull_first_available(EMBED_MODELS, "embed")
    print("\nModels ready. Set env vars before running Streamlit:")
    print("  set EMBED_BACKEND=ollama")
    print(f"  set OLLAMA_EMBED_MODEL={embed_model}")
    print(f"  set OLLAMA_MODEL={text_model}")
    print("\nThen run:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
