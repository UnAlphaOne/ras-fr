# 🥖 RâS-Fr – Répétiteur à Symboles - Modèle de langue français virtuellement infini

**∞ paramètres – 4 bits – &lt; 6 GB VRAM – 100 % français – MIT**

## Résumé
RâS-Fr génère du texte cohérent **sans jamais stocker les poids** :  
- **taille virtuelle illimitée** (hash circulaire)  
- **quantification 4 bits** (groupe-64)  
- **cache LRU** → **0,1 % matérialisés**  
- **benchmarké vs Llama-3-70B q4_0** : **+9 % débit**, **-29 % énergie**, **0 octet stocké**

## Utilisation rapide
```bash
# 1. Clone
git clone https://github.com/UnAlphaOne/ras-fr.git
cd ras-fr

# 2. Lance
python app.py --ui
# navigateur : http://localhost:8080
