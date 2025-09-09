#!/usr/bin/env python3
"""
Phrase Predictor Module
Maps gestures/blinks to meaningful phrases and adapts to user usage.
"""

import os, sqlite3, logging, time
from typing import List, Tuple, Optional
from collections import deque
from enum import Enum, auto
from contextlib import contextmanager

# ✅ Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PhrasePredictor")

# ✅ Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
PHRASE_DB = os.path.abspath(os.path.join(DATA_DIR, "phrases.db"))

# ✅ Prediction Methods Enum
class PredictionMethod(Enum):
    RULE_BASED = auto()
    FREQUENCY = auto()
    CONTEXT_SIMILARITY = auto()
    FALLBACK = auto()

# ✅ DB Context Manager
@contextmanager
def db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

# ✅ PhrasePredictor Class
class PhrasePredictor:
    def __init__(self):
        self.context_history = deque(maxlen=10)
        self.user_id = "default_user"
        self._init_db()
        self._load_cache()

    def _init_db(self):
        """Ensure DB tables exist and insert default phrases."""
        with db_connection(PHRASE_DB) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS phrases (
                                id INTEGER PRIMARY KEY,
                                text TEXT UNIQUE,
                                base_usage_count INTEGER DEFAULT 0
                            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS user_profiles (
                                user_id TEXT,
                                phrase_id INTEGER,
                                usage_count INTEGER DEFAULT 0,
                                last_used TIMESTAMP,
                                PRIMARY KEY(user_id, phrase_id)
                            )""")
            conn.commit()

            count = conn.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
            if count == 0:
                defaults = [
                    "hello", "how are you", "thank you", "yes", "no",
                    "please help me", "I need assistance", "good morning",
                    "good night", "I am okay", "can you hear me?"
                ]
                for p in defaults:
                    conn.execute("INSERT OR IGNORE INTO phrases (text, base_usage_count) VALUES (?,?)", (p, 1))
                conn.commit()
                logger.info("✅ Default phrases inserted")

    def _load_cache(self):
        with db_connection(PHRASE_DB) as conn:
            self.phrase_cache = [row[0] for row in conn.execute("SELECT text FROM phrases").fetchall()]

    # ✅ Gesture → Phrase Mapping + Frequency Based Suggestions
    def suggest_phrases(self, gesture: str = "", context: str = "") -> Tuple[List[str], PredictionMethod]:
        """
        Suggest phrases based on gesture or context.
        """
        try:
            gesture = gesture.lower() if gesture else ""

            # 🔹 Strong Rule-based Mapping
            gesture_map = {
                "open_palm": ["hello", "hi there", "good to see you"],
                "fist": ["stop", "wait a moment", "hold on"],
                "thumbs_up": ["yes", "I agree", "that's right"],
                "thumbs_down": ["no", "I disagree"],
                "pointing": ["look there", "that one", "check this"],
                "raised_hand": ["excuse me", "please wait", "I have a question"],
                "head_nod": ["yes", "absolutely", "I understand"],
                "head_shake": ["no", "not sure", "I disagree"],
                "double_blink": ["please help me", "I need assistance"]
            }

            if gesture in gesture_map:
                return gesture_map[gesture], PredictionMethod.RULE_BASED

            # 🔹 Frequency-based Fallback
            with db_connection(PHRASE_DB) as conn:
                rows = conn.execute("""
                    SELECT text FROM phrases ORDER BY base_usage_count DESC LIMIT 5
                """).fetchall()
                if rows:
                    return [r[0] for r in rows], PredictionMethod.FREQUENCY

            # 🔹 Hard fallback
            return ["hello", "yes", "no"], PredictionMethod.FALLBACK

        except Exception as e:
            logger.error(f"❌ Error in suggest_phrases: {e}")
            return ["hello"], PredictionMethod.FALLBACK

    # ✅ Record Phrase Usage
    def record_phrase_usage(self, phrase: str, gesture: Optional[str] = None):
        """Store usage stats for personalization."""
        try:
            with db_connection(PHRASE_DB) as conn:
                phrase_id = conn.execute("SELECT id FROM phrases WHERE text = ?", (phrase,)).fetchone()
                if not phrase_id:
                    conn.execute("INSERT INTO phrases (text, base_usage_count) VALUES (?,?)", (phrase, 1))
                    phrase_id = conn.execute("SELECT id FROM phrases WHERE text = ?", (phrase,)).fetchone()
                pid = phrase_id[0]

                conn.execute("""
                    INSERT INTO user_profiles (user_id, phrase_id, usage_count, last_used)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(user_id, phrase_id) DO UPDATE SET 
                        usage_count = usage_count + 1,
                        last_used = excluded.last_used
                """, (self.user_id, pid, time.time()))
                conn.commit()
            logger.info(f"📌 Usage recorded: {phrase} | Gesture: {gesture}")
        except Exception as e:
            logger.error(f"❌ Error recording phrase usage: {e}")
