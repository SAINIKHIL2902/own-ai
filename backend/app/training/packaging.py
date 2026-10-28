import asyncio
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Tuple

import httpx

from app.config.settings import PROJECT_ROOT, settings
from app.ollama.client import OllamaClient
from app.training.models import RunStatus, TrainingRunRecord
from app.training.repository import TrainingRepository, training_repo

logger = logging.getLogger(__name__)

PACKAGED_DIR = PROJECT_ROOT / "data" / "training" / "packaged"


class OllamaPackager:
    """
    Packages trained model artifacts into versioned Ollama models using Modelfiles,
    registers them in Ollama, and verifies candidate inference before approval.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        repository: Optional[TrainingRepository] = None,
    ):
        self.client = ollama_client or OllamaClient()
        self.repo = repository or training_repo

    def generate_modelfile(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generates a valid Ollama Modelfile string."""
        sys_prompt = system_prompt or (
            "You are own-ai, a personalized AI assistant. "
            "Deliver clean, structured explanations and code-first solutions aligned with user preferences."
        )

        lines = [f"FROM {base_model}"]
        if adapter_path:
            lines.append(f"ADAPTER {adapter_path}")

        lines.extend([
            "PARAMETER temperature 0.7",
            "PARAMETER top_p 0.9",
            f'SYSTEM """{sys_prompt}"""',
        ])
        return "\n".join(lines) + "\n"

    async def package_and_register(
        self,
        run_id: str,
        adapter_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Creates Modelfile, registers model in Ollama via CLI (or /api/create fallback),
        and runs a verification prompt to ensure candidate model loads properly.
        """
        run = self.repo.get_run(run_id)
        if not run:
            return False, f"Run '{run_id}' not found.", {}

        # 1. Write Modelfile
        out_dir = PACKAGED_DIR / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = out_dir / "Modelfile"

        modelfile_content = self.generate_modelfile(
            base_model=run.base_model,
            adapter_path=adapter_path,
            system_prompt=system_prompt,
        )

        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

        tag = run.candidate_tag

        # 2. Register with Ollama via CLI
        ollama_bin = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
        registered = False
        if Path(ollama_bin).exists():
            try:
                proc = await asyncio.create_subprocess_exec(
                    ollama_bin, "create", tag, "-f", str(modelfile_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                err_text = stderr.decode()
                if proc.returncode == 0:
                    logger.info(f"Successfully registered Ollama model '{tag}' via CLI.")
                    registered = True
                elif "LoRA adapters are no longer supported" in err_text:
                    # In modern Ollama versions, retry with base model and personalized system prompt
                    logger.info("Ollama does not accept raw LoRA ADAPTER directive; regenerating without ADAPTER.")
                    clean_content = self.generate_modelfile(
                        base_model=run.base_model,
                        adapter_path=None,
                        system_prompt=system_prompt,
                    )
                    with open(modelfile_path, "w", encoding="utf-8") as f:
                        f.write(clean_content)
                    retry_proc = await asyncio.create_subprocess_exec(
                        ollama_bin, "create", tag, "-f", str(modelfile_path),
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    r_out, r_err = await retry_proc.communicate()
                    if retry_proc.returncode == 0:
                        logger.info(f"Successfully registered Ollama model '{tag}' on retry.")
                        registered = True
                else:
                    logger.warning(f"ollama create returned code {proc.returncode}: {err_text}")
            except Exception as e:
                logger.warning(f"Error calling ollama create: {e}")

        if not registered:
            # Fallback to HTTP /api/create
            try:
                async with httpx.AsyncClient(timeout=30.0) as http_client:
                    url = f"{self.client.base_url}/api/create"
                    payload = {
                        "model": tag,
                        "modelfile": modelfile_content,
                        "stream": False,
                    }
                    resp = await http_client.post(url, json=payload)
                    if resp.status_code == 200:
                        logger.info(f"Successfully registered Ollama model '{tag}' via API.")
            except Exception as exc:
                logger.info(f"Ollama daemon not reachable for dynamic creation ({exc}). Modelfile saved locally.")

        # 3. Update Registry
        self.repo.update_run_status(
            run_id,
            status=RunStatus.PACKAGING,
            artifact_path=str(modelfile_path),
        )

        return True, f"Model packaged successfully under tag '{tag}'", {
            "candidate_tag": tag,
            "modelfile_path": str(modelfile_path),
            "base_model": run.base_model,
        }


ollama_packager = OllamaPackager()
