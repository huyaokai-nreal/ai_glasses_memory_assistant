from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest


cryptography = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


TOOL_PATH = Path(__file__).resolve().parents[1] / "android" / "tools" / "DecryptDiagnosticBundle.java"


def _encrypted_fixture(plaintext: bytes, password: str) -> bytes:
    salt = bytes(range(16))
    iv = bytes(range(12))
    iterations = 210_000
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=32)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return b"".join(
        (
            b"AIGDIAG1",
            struct.pack(">I", iterations),
            bytes((len(salt),)),
            salt,
            bytes((len(iv),)),
            iv,
            ciphertext,
            encryptor.tag,
        )
    )


def test_java_diagnostic_tool_decrypts_android_format_and_refuses_overwrite(tmp_path) -> None:
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin" / "java") if java_home else shutil.which("java")
    if java is None:
        pytest.skip("Java source launcher is unavailable")
    version = subprocess.run([java, "-version"], capture_output=True, check=False)
    if version.returncode != 0:
        pytest.skip("Java source launcher is unavailable")
    source = tmp_path / "report.aigd"
    destination = tmp_path / "report.zip"
    plaintext = b"PK\x03\x04diagnostics"
    source.write_bytes(_encrypted_fixture(plaintext, "correct-password"))
    command = [java, str(TOOL_PATH), str(source), "--output", str(destination)]

    completed = subprocess.run(command, input="correct-password\n", text=True, capture_output=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert destination.read_bytes() == plaintext
    repeated = subprocess.run(command, input="correct-password\n", text=True, capture_output=True, check=False)
    assert repeated.returncode == 1
    assert "refusing to overwrite" in repeated.stderr
    assert destination.read_bytes() == plaintext
