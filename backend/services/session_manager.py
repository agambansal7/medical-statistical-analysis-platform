"""Session management for analysis sessions."""

import uuid
import json
import os
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path
import pandas as pd
from threading import Lock
import pickle
import hashlib


class AnalysisSession:
    """Represents a single analysis session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.data: Optional[pd.DataFrame] = None
        self.data_profile: Optional[Dict[str, Any]] = None
        self.filename: Optional[str] = None
        self.analyses: List[Dict[str, Any]] = []
        self.chat_history: List[Dict[str, Any]] = []
        self.analysis_plan: Optional[Dict[str, Any]] = None
        self.figures: List[str] = []
        self.results_report: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "data_loaded": self.data is not None,
            "filename": self.filename,
            "n_analyses_performed": len(self.analyses),
            "n_figures": len(self.figures),
        }


class SessionManager:
    """Manage multiple analysis sessions."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions: Dict[str, AnalysisSession] = {}
        return cls._instance

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = AnalysisSession(session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return [session.to_dict() for session in self._sessions.values()]

    def set_data(self, session_id: str, data: pd.DataFrame,
                 filename: str, profile: Dict[str, Any]) -> bool:
        """Set data for a session."""
        session = self.get_session(session_id)
        if session:
            session.data = data
            session.filename = filename
            session.data_profile = profile
            return True
        return False

    def add_analysis(self, session_id: str, analysis: Dict[str, Any]) -> bool:
        """Add analysis result to session."""
        session = self.get_session(session_id)
        if session:
            session.analyses.append({
                **analysis,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False

    def add_chat_message(self, session_id: str, role: str, content: str) -> bool:
        """Add chat message to session history."""
        session = self.get_session(session_id)
        if session:
            session.chat_history.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            return True
        return False

    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat history for session."""
        session = self.get_session(session_id)
        if session:
            return session.chat_history
        return []

    def add_figure(self, session_id: str, figure_path: str) -> bool:
        """Add figure path to session."""
        session = self.get_session(session_id)
        if session:
            session.figures.append(figure_path)
            return True
        return False

    def save_session(self, session_id: str, save_dir: str = None) -> Dict[str, Any]:
        """Save session to disk for later restoration.

        Args:
            session_id: The session to save
            save_dir: Directory to save session files (default: ./saved_sessions)

        Returns:
            Dictionary with save path and session info
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        # Default save directory
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_sessions")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"session_{session_id[:8]}_{timestamp}"

        # Save metadata as JSON
        metadata = {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "saved_at": datetime.now().isoformat(),
            "filename": session.filename,
            "n_rows": len(session.data) if session.data is not None else 0,
            "n_cols": len(session.data.columns) if session.data is not None else 0,
            "columns": list(session.data.columns) if session.data is not None else [],
            "n_analyses": len(session.analyses),
            "n_chat_messages": len(session.chat_history),
            "has_analysis_plan": session.analysis_plan is not None,
            "has_report": session.results_report is not None
        }

        metadata_path = os.path.join(save_dir, f"{filename_base}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save data as CSV (if exists)
        data_path = None
        if session.data is not None:
            data_path = os.path.join(save_dir, f"{filename_base}_data.csv")
            session.data.to_csv(data_path, index=False)

        # Save session state (analyses, chat history, etc.) as pickle
        state = {
            "analyses": session.analyses,
            "chat_history": session.chat_history,
            "analysis_plan": session.analysis_plan,
            "data_profile": session.data_profile,
            "figures": session.figures,
            "results_report": session.results_report
        }
        state_path = os.path.join(save_dir, f"{filename_base}_state.pkl")
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)

        return {
            "success": True,
            "session_id": session_id,
            "metadata_path": metadata_path,
            "data_path": data_path,
            "state_path": state_path,
            "save_name": filename_base,
            "metadata": metadata
        }

    def load_session(self, save_name: str, save_dir: str = None) -> Dict[str, Any]:
        """Load a previously saved session.

        Args:
            save_name: The base name of the saved session files
            save_dir: Directory containing saved session files

        Returns:
            Dictionary with new session_id and status
        """
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_sessions")

        metadata_path = os.path.join(save_dir, f"{save_name}_metadata.json")
        data_path = os.path.join(save_dir, f"{save_name}_data.csv")
        state_path = os.path.join(save_dir, f"{save_name}_state.pkl")

        if not os.path.exists(metadata_path):
            return {"success": False, "error": f"Session files not found: {save_name}"}

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create new session
        new_session_id = self.create_session()
        session = self.get_session(new_session_id)

        # Load data if exists
        if os.path.exists(data_path):
            session.data = pd.read_csv(data_path)
            session.filename = metadata.get("filename")

        # Load state
        if os.path.exists(state_path):
            with open(state_path, 'rb') as f:
                state = pickle.load(f)

            session.analyses = state.get("analyses", [])
            session.chat_history = state.get("chat_history", [])
            session.analysis_plan = state.get("analysis_plan")
            session.data_profile = state.get("data_profile")
            session.figures = state.get("figures", [])
            session.results_report = state.get("results_report")

        return {
            "success": True,
            "new_session_id": new_session_id,
            "original_session_id": metadata.get("session_id"),
            "loaded_at": datetime.now().isoformat(),
            "metadata": metadata
        }

    def list_saved_sessions(self, save_dir: str = None) -> List[Dict[str, Any]]:
        """List all saved sessions.

        Args:
            save_dir: Directory containing saved sessions

        Returns:
            List of saved session metadata
        """
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "saved_sessions")

        if not os.path.exists(save_dir):
            return []

        saved_sessions = []
        for filename in os.listdir(save_dir):
            if filename.endswith("_metadata.json"):
                filepath = os.path.join(save_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        metadata = json.load(f)
                    metadata["save_name"] = filename.replace("_metadata.json", "")
                    saved_sessions.append(metadata)
                except Exception:
                    pass

        # Sort by saved_at descending
        saved_sessions.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return saved_sessions

    def duplicate_session(self, session_id: str) -> Dict[str, Any]:
        """Create a copy of an existing session.

        Args:
            session_id: Session to duplicate

        Returns:
            Dictionary with new session_id
        """
        source = self.get_session(session_id)
        if not source:
            return {"success": False, "error": "Session not found"}

        # Create new session
        new_session_id = self.create_session()
        new_session = self.get_session(new_session_id)

        # Copy data
        new_session.data = source.data.copy() if source.data is not None else None
        new_session.data_profile = source.data_profile.copy() if source.data_profile else None
        new_session.filename = source.filename

        # Copy analyses (deep copy)
        import copy
        new_session.analyses = copy.deepcopy(source.analyses)
        new_session.chat_history = copy.deepcopy(source.chat_history)
        new_session.analysis_plan = copy.deepcopy(source.analysis_plan) if source.analysis_plan else None
        new_session.figures = source.figures.copy()
        new_session.results_report = source.results_report

        return {
            "success": True,
            "new_session_id": new_session_id,
            "source_session_id": session_id
        }

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get detailed summary of a session.

        Args:
            session_id: Session to summarize

        Returns:
            Detailed session summary
        """
        session = self.get_session(session_id)
        if not session:
            return {"success": False, "error": "Session not found"}

        # Summarize analyses by type
        analysis_summary = {}
        for analysis in session.analyses:
            atype = analysis.get("analysis_type", "unknown")
            if atype not in analysis_summary:
                analysis_summary[atype] = 0
            analysis_summary[atype] += 1

        return {
            "success": True,
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "data": {
                "loaded": session.data is not None,
                "filename": session.filename,
                "n_rows": len(session.data) if session.data is not None else 0,
                "n_columns": len(session.data.columns) if session.data is not None else 0,
                "columns": list(session.data.columns) if session.data is not None else []
            },
            "analyses": {
                "total": len(session.analyses),
                "by_type": analysis_summary,
                "latest": session.analyses[-1] if session.analyses else None
            },
            "chat": {
                "n_messages": len(session.chat_history),
                "last_message": session.chat_history[-1] if session.chat_history else None
            },
            "plan": {
                "has_plan": session.analysis_plan is not None,
                "confirmed": session.analysis_plan.get("confirmed", False) if session.analysis_plan else False
            },
            "report": {
                "generated": session.results_report is not None
            }
        }


# Global session manager instance
session_manager = SessionManager()
