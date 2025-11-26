from pydantic import BaseModel, Field, field_validator

from backend.utils.clean_text import clean_text

# The list of categories must be in the exact order that your model returns them
OUTPUT_CATEGORIES = [
    'IsToxic', 
    'IsAbusive', 
    'IsProvocative', 
    'IsObscene', 
    'IsHatespeech', 
    'IsRacist', 
    'IsThreat', 
    'IsReligiousHate', 
    'IsNationalist'
]

# Define the input structure
class TextIn(BaseModel):
    """The structure for the incoming request."""
    text: str = Field(
        ..., 
        example="You should be banned from the internet for saying that.",
        description="The text to classify for toxicity."
    )
    
    @field_validator('text', mode='before')
    @classmethod
    def apply_preprocessing(cls, value):
        """
        Uses the external clean_text function to preprocess the input 
        before Pydantic validates it.
        """
        # Call the imported function with the raw input value
        cleaned_value = clean_text(value)
        return cleaned_value

# Define the output structure (matching your 10 categories)
# All models will use this structure for consistency.
class ToxicityOut(BaseModel):
    """The structure for the prediction response."""
    IsToxic: float = Field(..., ge=0.0, le=1.0)
    IsAbusive: float = Field(..., ge=0.0, le=1.0)
    IsProvocative: float = Field(..., ge=0.0, le=1.0)
    IsObscene: float = Field(..., ge=0.0, le=1.0)
    IsHatespeech: float = Field(..., ge=0.0, le=1.0)
    IsRacist: float = Field(..., ge=0.0, le=1.0)
    IsThreat: float = Field(..., ge=0.0, le=1.0)
    IsReligiousHate: float = Field(..., ge=0.0, le=1.0)
    IsNationalist: float = Field(..., ge=0.0, le=1.0)