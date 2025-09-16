#!/usr/bin/env python3
"""
Script de inicio del sistema completo - Frontend + Backend + API
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import threading

def run_backend_api():
    """Ejecutar la API backend"""
    print("🚀 Iniciando API Backend...")
    os.chdir(Path(__file__).parent / "api")
    subprocess.run([sys.executable, "main_api.py"])

def run_frontend():
    """Ejecutar el frontend React"""
    print("🎨 Iniciando Frontend React...")
    frontend_dir = Path(__file__).parent / "frontend"
    try:
        if not (frontend_dir / "node_modules").exists():
            print("📦 Instalando dependencias de Node.js...")
            subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
        subprocess.run(["npm", "start"], cwd=frontend_dir, shell=True)
    except FileNotFoundError:
        print("❌ npm no encontrado. Frontend no disponible.")
        print("📡 Backend funcionando en: http://localhost:8000")
    except Exception as e:
        print(f"❌ Error iniciando frontend: {e}")
        print("📡 Backend funcionando en: http://localhost:8000")

def main():
    print("🤖 Sistema Multi-Agente Financiero - Inicio Completo")
    print("="*60)

    # Crear threads para backend y frontend
    backend_thread = threading.Thread(target=run_backend_api, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)

    # Iniciar backend primero
    backend_thread.start()
    time.sleep(3)  # Esperar a que el backend se inicie

    # Iniciar frontend
    frontend_thread.start()

    print("\n✅ Sistema iniciado:")
    print("   📡 API Backend: http://localhost:8000")
    print("   🎨 Frontend: http://localhost:3000")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("\nPresiona Ctrl+C para detener el sistema")

    try:
        # Mantener el script principal vivo
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo sistema...")
        sys.exit(0)

if __name__ == "__main__":
    main()
