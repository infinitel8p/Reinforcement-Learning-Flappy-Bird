Create a virtual environment with:
```bash
python -m venv venv
```

Start the virtual environment with:  
On Linux/MacOS:
```bash
source venv/bin/activate && cd v3
```

On Windows:
```bash
venv\Scripts\activate && cd v3
```

Install Python dependencies with:
```bash
pip install -r requirements.txt
```

Start training the model with:
```bash
python main.py cartpole --train
```

The model will train for unlimited episodes until it reaches a reward of 100000. After that you can test the model with:
```bash
python main.py cartpole
```