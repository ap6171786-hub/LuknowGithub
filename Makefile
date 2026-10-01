.PHONY: install run-api run-ui test clean

install:
	python -m pip install -r requirements.txt

run-api:
	python -m uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	python -m streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501

test:
	python -m pytest -q

clean:
	python -c "from pathlib import Path; import shutil; root=Path('.'); [shutil.rmtree(path, ignore_errors=True) for path in root.rglob('__pycache__')]; shutil.rmtree(root/'.pytest_cache', ignore_errors=True); [path.unlink(missing_ok=True) for path in root.glob('.coverage')]"

