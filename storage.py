import os, json, base64, hashlib
from cryptography.fernet import Fernet
from dataclasses import dataclass, asdict

FERNET_KEY = os.getenv("FERNET_KEY", Fernet.generate_key().decode())
fernet = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "users.enc")
os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

@dataclass
class User:
    email: str
    birth_date: str
    birth_time: str
    birth_place: str
    latitude: float
    longitude: float
    timezone: str

def hash_email(email):
    return hashlib.sha256(email.encode()).hexdigest()

def load_store():
    if not os.path.exists(DATA_FILE):
        return {}
    with open(DATA_FILE, 'rb') as f:
        decrypted = fernet.decrypt(f.read()).decode()
        return {k: User(**v) for k, v in json.loads(decrypted).items()}

def save_store(store):
    serializable = {k: asdict(v) for k, v in store.items()}
    encrypted = fernet.encrypt(json.dumps(serializable).encode())
    with open(DATA_FILE, 'wb') as f:
        f.write(encrypted)
