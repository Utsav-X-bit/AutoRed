import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List

class KnowledgeBase:
    def __init__(self, db_path: str = "data/autored_kb.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trajectories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    timestamp DATETIME,
                    scenario_id TEXT,
                    defense_prompt TEXT,
                    ground_truth TEXT,
                    state_id TEXT,
                    chosen_strategy TEXT,
                    alternative_strategies TEXT,
                    decision_reason TEXT,
                    decision_confidence FLOAT,
                    planner_thoughts TEXT,
                    attack_plan TEXT,
                    primitive_sequence TEXT,
                    generator_prompt TEXT,
                    attack_string TEXT,
                    victim_response TEXT,
                    extractor_candidates TEXT,
                    verifier_success BOOLEAN,
                    reward FLOAT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_snapshots (
                    state_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    attempt INTEGER,
                    state_json TEXT,
                    hash TEXT,
                    timestamp DATETIME
                )
            ''')
            # Add an index for faster RAG queries later
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_scenario ON trajectories(scenario_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_success ON trajectories(verifier_success)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_state ON state_snapshots(hash)')
            
            # Simple schema migration
            try:
                cursor.execute('ALTER TABLE trajectories ADD COLUMN state_id TEXT')
                cursor.execute('ALTER TABLE trajectories ADD COLUMN chosen_strategy TEXT')
                cursor.execute('ALTER TABLE trajectories ADD COLUMN alternative_strategies TEXT')
                cursor.execute('ALTER TABLE trajectories ADD COLUMN decision_reason TEXT')
                cursor.execute('ALTER TABLE trajectories ADD COLUMN decision_confidence FLOAT')
            except sqlite3.OperationalError:
                pass  # Columns exist
                
            conn.commit()

    def log_trajectory(self, data: Dict[str, Any]):
        """Logs a single step trajectory into the DB."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Log State Snapshot if provided
            state_data = data.get("state_snapshot")
            state_id = ""
            if state_data:
                state_id = state_data.get("state_id", "")
                cursor.execute('''
                    INSERT OR IGNORE INTO state_snapshots (
                        state_id, session_id, attempt, state_json, hash, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    state_id,
                    data.get("session_id", "default"),
                    state_data.get("attempt", 0),
                    json.dumps(state_data.get("state_json", {})),
                    state_data.get("hash", ""),
                    datetime.now().isoformat()
                ))

            # 2. Log Trajectory
            cursor.execute('''
                INSERT INTO trajectories (
                    session_id, timestamp, scenario_id, defense_prompt, ground_truth,
                    state_id, chosen_strategy, alternative_strategies, decision_reason, decision_confidence,
                    planner_thoughts, attack_plan, primitive_sequence, generator_prompt,
                    attack_string, victim_response, extractor_candidates,
                    verifier_success, reward
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.get("session_id", "default"),
                datetime.now().isoformat(),
                data.get("scenario_id", ""),
                data.get("defense_prompt", ""),
                data.get("ground_truth", ""),
                state_id,
                data.get("chosen_strategy", ""),
                json.dumps(data.get("alternative_strategies", [])),
                data.get("decision_reason", ""),
                data.get("decision_confidence", 0.0),
                data.get("planner_thoughts", ""),
                data.get("attack_plan", ""),
                data.get("primitive_sequence", ""),
                data.get("generator_prompt", ""),
                data.get("attack_string", ""),
                data.get("victim_response", ""),
                json.dumps(data.get("extractor_candidates", [])),
                data.get("verifier_success", False),
                data.get("reward", 0.0)
            ))
            conn.commit()
            
    def get_successful_trajectories(self, scenario_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Used for RAG later to find successful attacks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM trajectories WHERE verifier_success = 1"
            params = []
            if scenario_id:
                query += " AND scenario_id = ?"
                params.append(scenario_id)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
