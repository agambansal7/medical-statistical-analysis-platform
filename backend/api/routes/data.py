"""Data upload and management endpoints."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
from typing import Optional

from services import session_manager, analysis_service
from models import DataUploadResponse, DataProfile
from core.config import settings

router = APIRouter(prefix="/data", tags=["Data"])


@router.post("/upload", response_model=DataUploadResponse)
async def upload_data(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Upload a data file (CSV or Excel).

    Creates a new session if session_id is not provided.
    """
    # Validate file extension
    filename = file.filename.lower()
    valid_extensions = settings.ALLOWED_EXTENSIONS

    if not any(filename.endswith(ext) for ext in valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {valid_extensions}"
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_MB}MB"
        )

    # Create or get session
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            session_id = session_manager.create_session()
    else:
        session_id = session_manager.create_session()

    # Parse file
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:  # Excel
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file: {str(e)}"
        )

    # Profile data
    profile = analysis_service.profile_data(df)

    # Store in session
    session_manager.set_data(session_id, df, file.filename, profile)

    # Get preview
    preview = df.head(10).to_dict(orient="records")

    # Build variables list for response
    variables_list = []
    for var in profile.get("variables", []):
        variables_list.append({
            "name": var.get("name"),
            "statistical_type": var.get("statistical_type"),
            "n_unique": var.get("n_unique"),
            "n_missing": var.get("n_missing", 0),
            "missing_pct": var.get("missing_pct", 0.0),
        })

    return DataUploadResponse(
        success=True,
        session_id=session_id,
        message=f"Successfully loaded {len(df)} rows and {len(df.columns)} columns",
        profile=DataProfile(
            session_id=session_id,
            filename=file.filename,
            n_rows=profile["n_rows"],
            n_columns=profile["n_columns"],
            n_continuous=profile["n_continuous"],
            n_categorical=profile["n_categorical"],
            n_binary=profile["n_binary"],
            variables=variables_list,
            potential_outcomes=profile.get("potential_outcome_columns", []),
            potential_groups=profile.get("potential_group_columns", []),
            warnings=profile.get("warnings", [])
        ),
        preview=preview
    )


@router.get("/profile/{session_id}")
async def get_data_profile(session_id: str):
    """Get detailed data profile for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data_profile is None:
        raise HTTPException(status_code=400, detail="No data loaded in session")

    return JSONResponse(content={
        "success": True,
        "profile": session.data_profile
    })


@router.get("/preview/{session_id}")
async def get_data_preview(session_id: str, n_rows: int = 20):
    """Get data preview for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded in session")

    preview = session.data.head(n_rows).to_dict(orient="records")

    return JSONResponse(content={
        "success": True,
        "data": preview,
        "columns": list(session.data.columns),
        "dtypes": {col: str(dtype) for col, dtype in session.data.dtypes.items()}
    })


@router.get("/columns/{session_id}")
async def get_columns(session_id: str):
    """Get column information for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    columns = []
    for var in session.data_profile.get("variables", []):
        columns.append({
            "name": var["name"],
            "type": var["statistical_type"],
            "n_missing": var["n_missing"],
            "n_unique": var["n_unique"]
        })

    return JSONResponse(content={
        "success": True,
        "columns": columns
    })


@router.get("/describe/{session_id}/{column}")
async def describe_column(session_id: str, column: str):
    """Get detailed description of a specific column."""
    session = session_manager.get_session(session_id)
    if not session or session.data is None:
        raise HTTPException(status_code=404, detail="Session or data not found")

    if column not in session.data.columns:
        raise HTTPException(status_code=404, detail=f"Column '{column}' not found")

    col_data = session.data[column]

    # Find variable profile
    var_profile = None
    for var in session.data_profile.get("variables", []):
        if var["name"] == column:
            var_profile = var
            break

    return JSONResponse(content={
        "success": True,
        "column": column,
        "profile": var_profile,
        "sample_values": col_data.dropna().head(10).tolist()
    })
