"""
Section-specific prompts for IQ test generation
Each section is generated individually to avoid token limits
"""

def get_section_prompts(age: int = 14) -> dict:
    """Get all section prompts with age placeholders replaced"""
    age_str = str(age)
    
    prompts = {
        "section_1": f"""
You are an expert psychometric test designer specializing in cognitive assessment for adolescents. Generate a comprehensive IQ test suitable for {age}-year-old test takers following these exact specifications below:

Section: Verbal Reasoning (10-12 questions, 7-8 minutes)
Format: Multiple choice (A, B, C, D). Choose the best answer that completes or explains the statement.
Generate questions testing:
- Vocabulary (word definitions, synonyms/antonyms)
- Verbal analogies (word relationships)
- Sentence completion (logical word choice)
- Reading comprehension (short passages with inference questions)
- Verbal classification (identifying word that doesn't belong)

Output Format:
- Question: Question text
- Options: List of options (A, B, C, D)
- Answer: Correct answer
""",
        
        "section_2": f"""
You are an expert psychometric test designer specializing in cognitive assessment for adolescents. Generate a comprehensive IQ test suitable for {age}-year-old test takers following these exact specifications below:

Section: Mathematical Reasoning (10-12 questions, 10 minutes)
Format: Multiple choice (A, B, C, D). Choose the best answer that completes or explains the statement.
Generate questions testing:
- Number sequences and patterns
- Word problems (practical math scenarios)
- Proportions, percentages & Ratios
- Logical deduction (if-then scenarios)
- Coding/decoding
- Numerical Analogies (Identifying relationships between number pairs)
- Unit Conversion & Measurement (Problems involving currency, time, length, or volume conversions)

Output Format:
- Question: Question text
- Options: List of options (A, B, C, D)
- Answer: Correct answer
""",
        
        "section_3": f"""
You are an expert psychometric test designer specializing in cognitive assessment for adolescents. Generate a comprehensive IQ test suitable for {age}-year-old test takers following these exact specifications below:

Section: Spatial/Visual Reasoning (10-12 questions, 7-8 minutes)
Format: Multiple choice (A, B, C, D). Choose the best answer that completes or explains the statement.
Generate questions testing:
- Pattern completion (2D patterns in grids)
- Mental rotation (which shape matches when rotated)
- Spatial folding (how shapes look when folded/unfolded)
- Visual analogies (shape relationships)
- Matrix reasoning (nxm grids with missing element)
- Hidden figures (identifying shapes within complex figures)

Output Format:
- Question: Question text (Describe patterns clearly using text, symbols, or ASCII art)
- Options: List of options (A, B, C, D)
- Answer: Correct answer
"""
    }
    
    return prompts
