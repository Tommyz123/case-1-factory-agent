"""
Data models for the Factory Search Agent
All models use Pydantic for validation
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class CompanyInput(BaseModel):
    """Input model for companies to search"""
    companies: List[str] = Field(
        ...,
        description="List of company names to search for factories",
        min_length=1
    )


class SearchResult(BaseModel):
    """Model for raw search results"""
    url: str = Field(..., description="URL of the search result")
    title: str = Field(..., description="Title of the search result")
    content: str = Field(..., description="Content/snippet from the search result")


class FacilityCandidate(BaseModel):
    """Model for facility candidates extracted from search results"""
    company_name: str = Field(..., description="Name of the company")
    facility_name: str = Field(..., description="Name of the manufacturing facility")
    facility_type: str = Field(
        ...,
        description="Type of facility (factory, plant, refinery, assembly, etc.)"
    )
    full_address: str = Field(..., description="Complete address of the facility")
    country: str = Field(..., description="Country where the facility is located")
    city: str = Field(..., description="City where the facility is located")
    latitude: Optional[float] = Field(
        None,
        description="Latitude coordinate (optional)",
        ge=-90,
        le=90
    )
    longitude: Optional[float] = Field(
        None,
        description="Longitude coordinate (optional)",
        ge=-180,
        le=180
    )
    evidence_urls: List[str] = Field(
        ...,
        description="URLs that provide evidence for this facility (MANDATORY)",
        min_length=1  # Critical: must have at least one evidence URL
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Confidence score for this extraction (0-1)",
        ge=0.0,
        le=1.0
    )


class ValidatedFacility(FacilityCandidate):
    """Model for validated facilities (extends FacilityCandidate)"""
    validation_passed: bool = Field(
        ...,
        description="Whether the facility passed validation"
    )
    validation_reason: str = Field(
        ...,
        description="Reason for validation pass/fail"
    )


class FinalOutput(BaseModel):
    """Final output model containing all validated facilities"""
    facilities: List[ValidatedFacility] = Field(
        default_factory=list,
        description="List of validated manufacturing facilities"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the search process"
    )


# Type definitions for LangGraph state
class AgentState(BaseModel):
    """State model for the LangGraph workflow"""
    company: str = Field(..., description="Current company being processed")
    search_results: List[SearchResult] = Field(
        default_factory=list,
        description="Raw search results"
    )
    extracted_facilities: List[FacilityCandidate] = Field(
        default_factory=list,
        description="Facilities extracted from search results"
    )
    validated_facilities: List[ValidatedFacility] = Field(
        default_factory=list,
        description="Facilities that passed validation"
    )
    final_output: Optional[FinalOutput] = Field(
        None,
        description="Final output for this company"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any errors encountered during processing"
    )
