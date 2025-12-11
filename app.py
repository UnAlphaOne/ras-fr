
---

#### 3. `app.py` – **API REST**
```python
# app.py – API REST RâS-Fr v7.0 – 100 % français
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch, time
from v4_infini import generer_infini
from transformers import CamembertTokenizer

app = FastAPI(title="RâS-Fr v7.0 API", version="7.0")

class Corps(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 1.0

class Reponse(BaseModel):
    texte: str
    tok_s: float
    vram_gb: float
    watts: float

@app.post("/generer", response_model=Reponse)
def generer_endpoint(corps: Corps):
    if len(corps.prompt) > 512:
        raise HTTPException(status_code=400, detail="Prompt trop long (max 512).")
    torch.cuda.synchronize()
    t0 = time.time()
    texte = generer_infini(corps.prompt, max_tokens=corps.max_tokens, temperature=corps.temperature)
    torch.cuda.synchronize()
    tok_s = corps.max_tokens / (time.time() - t0)
    vram = torch.cuda.max_memory_allocated() / 1e9
    import pynvml
    pynvml.nvmlInit()
    watts = pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(0)) / 1000
    return Reponse(texte=texte, tok_s=round(tok_s, 1), vram_gb=round(vram, 2), watts=round(watts, 1))

@app.get("/")
def racine():
    return {"message": "Bienvenue sur RâS-Fr v7.0 – API 100 % française"}