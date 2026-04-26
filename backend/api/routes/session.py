"""Session management endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from services import session_manager

router = APIRouter(prefix="/session", tags=["Session"])


@router.post("/create")
async def create_session():
    """Create a new analysis session."""
    session_id = session_manager.create_session()
    return JSONResponse(content={
        "success": True,
        "session_id": session_id,
        "message": "Session created successfully"
    })


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content={
        "success": True,
        "session": session.to_dict()
    })


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content={
        "success": True,
        "message": "Session deleted"
    })


@router.get("/")
async def list_sessions():
    """List all active sessions."""
    sessions = session_manager.list_sessions()
    return JSONResponse(content={
        "success": True,
        "sessions": sessions
    })


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get comprehensive session summary."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    summary = {
        "session_id": session_id,
        "created_at": session.created_at.isoformat(),
        "data": {
            "loaded": session.data is not None,
            "filename": session.filename,
            "n_rows": len(session.data) if session.data is not None else 0,
            "n_columns": len(session.data.columns) if session.data is not None else 0
        },
        "analyses": {
            "count": len(session.analyses),
            "types": list(set(a.get("analysis_type") for a in session.analyses))
        },
        "chat": {
            "message_count": len(session.chat_history)
        },
        "figures": {
            "count": len(session.figures)
        }
    }

    if session.analysis_plan:
        summary["analysis_plan"] = {
            "research_question": session.analysis_plan.get("research_question"),
            "research_type": session.analysis_plan.get("research_type"),
            "n_primary_analyses": len(session.analysis_plan.get("primary_analyses", [])),
            "n_secondary_analyses": len(session.analysis_plan.get("secondary_analyses", []))
        }

    return JSONResponse(content={
        "success": True,
        "summary": summary
    })


@router.post("/{session_id}/save")
async def save_session(session_id: str):
    """Save session to disk for later restoration.

    Saves the session data, analyses, chat history, and all state
    to disk. Returns a save_name that can be used to load the session later.
    """
    result = session_manager.save_session(session_id)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return JSONResponse(content=result)


@router.post("/load/{save_name}")
async def load_session(save_name: str):
    """Load a previously saved session.

    Creates a new session with all data and state from the saved session.
    Returns the new session_id.
    """
    result = session_manager.load_session(save_name)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return JSONResponse(content=result)


@router.get("/saved/list")
async def list_saved_sessions():
    """List all saved sessions available for loading."""
    saved = session_manager.list_saved_sessions()
    return JSONResponse(content={
        "success": True,
        "saved_sessions": saved
    })


@router.post("/{session_id}/duplicate")
async def duplicate_session(session_id: str):
    """Create a copy of an existing session.

    Creates a new session with all data, analyses, and state
    copied from the source session.
    """
    result = session_manager.duplicate_session(session_id)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return JSONResponse(content=result)


@router.get("/{session_id}/detailed-summary")
async def get_detailed_session_summary(session_id: str):
    """Get detailed summary of a session including analysis breakdown."""
    result = session_manager.get_session_summary(session_id)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))

    return JSONResponse(content=result)
