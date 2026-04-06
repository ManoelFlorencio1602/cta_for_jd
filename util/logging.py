import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.FileHandler("run.log"),   # salva no arquivo
        logging.StreamHandler(sys.stdout) # também mostra no terminal
    ]
)