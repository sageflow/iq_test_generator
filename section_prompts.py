"""
Section-specific prompts for IQ test generation
Each section is generated individually to avoid token limits
"""

def get_section_prompts(age: int = 14) -> dict:
    """Get all section prompts with age placeholders replaced"""
    age_str = str(age)
    
    prompts = {
        "section_1": f"""
You are an expert psychometric test designer specializing in cognitive assessment for adolescents. Generate a Verbal Reasoning section of an IQ test tailored for {age}-year-old test takers. The section must strictly adhere to the specifications below:

Section: Verbal Reasoning  
Length: 10 multiple-choice questions  
Time Limit: 7 minutes  
Format: Each question must have exactly four options labeled A, B, C, and D. Select the single best answer that logically completes the statement, solves the analogy, or correctly responds to the prompt.

Content Requirements:  
Include questions that assess the following core verbal reasoning skills, with age-appropriate vocabulary and complexity:  
- Vocabulary: Identify correct definitions, synonyms, or antonyms of moderately challenging but developmentally appropriate words.  
- Verbal Analogies: Recognize and apply semantic or logical relationships between word pairs (e.g., “doctor : hospital :: teacher : ___”).  
- Sentence Completion: Choose the word or phrase that best fits the context, maintaining grammatical correctness and logical coherence.  
- Reading Comprehension: Analyze short passages (1-3 sentences) and answer inference-based or main-idea questions.  
- Verbal Classification: Identify the word that does not belong in a given set based on category, function, or meaning.

Guidelines:  
- Use clear, standard English with no regional slang or culturally specific references.  
- Ensure vocabulary aligns with typical language development for the specified age group (e.g., avoid overly rare or technical terms unless contextually supported).  
- Maintain a balanced distribution—include approximately 2 questions per skill area.  
- All items must be original, unambiguous, and grounded in established cognitive assessment principles.

Output Format:  
- Begin with: Section Name: Verbal Reasoning (10 questions, 7 minutes)  
- For each question, provide:  
  Question: [Full question text]  
  Options:  
  A) [Option text]  
  B) [Option text]  
  C) [Option text]  
  D) [Option text]  
- Answer Key:  
  1. [Correct Option]  
  2. [Correct Option]  
  ... 
  10. [Correct Option]

Example Output:

Section 1: Verbal Reasoning (10 questions, 7 minutes)
1. Question: Choose the word that means the opposite of "ABUNDANT"
A) Scarce
B) Plentiful
C) Excessive
D) Numerous

2. Question: Complete the analogy: Doctor is to hospital as teacher is to ______
A) Classroom
B) Student
C) Book
D) Principal

3. Question: Choose the word that best completes the sentence: The scientist conducted the experiment with great ______ to ensure accurate results.
A) Carelessness
B) Precision
C) Haste
D) Confusion

4. Question: Read the passage: "The ancient civilization developed sophisticated irrigation systems that allowed them to farm in arid regions. Their success depended on careful water management and community cooperation." What was key to their agricultural success?
A) Modern technology
B) Water management and cooperation
C) Large workforce
D) Fertile soil

5. Question: Which word does not belong with the others?
A) Novel
B) Poetry
C) Mathematics
D) Biography

6. Question: Choose the synonym for "PERSEVERE"
A) Give up
B) Continue
C) Hesitate
D) Forget

7. Question: Complete the analogy: Water is to thirst as food is to ______
A) Hunger
B) Cooking
C) Eating
D) Nutrition

8. Question: Which word is spelled correctly?
A) Accomodate
B) Separate
C) Definately
D) Calender

9. Question: Choose the word that best fits: The detective needed ______ evidence to solve the complex case.
A) Circumstantial
B) Emotional
C) Imaginary
D) Temporary

10. Question: Which word does not belong?
A) Mercury
B) Venus
C) Jupiter
D) Galaxy

Answer Key:
1. A
2. A
3. B
4. B
5. C
6. B
7. A
8. B
9. A
10. D

""",
        
        "section_2": f"""
You are an expert psychometric test designer specializing in cognitive assessment for adolescents. Generate a Mathematical Reasoning section of an IQ test tailored for {age}-year-old test takers. The section must strictly adhere to the specifications below:

Section: Mathematical Reasoning  
Length: 10 multiple-choice questions  
Time Limit: 10 minutes  
Format: Each question must have exactly four options labeled A, B, C, and D. Select the single best answer that correctly solves the problem or completes the pattern.

Content Requirements:  
Include questions that assess the following mathematical reasoning skills, with age-appropriate complexity:  
- Number sequences and patterns (e.g., identifying the next number in a logical series)  
- Word problems involving realistic, everyday scenarios (e.g., shopping, travel, or scheduling)  
- Proportions, percentages, and ratios (e.g., calculating discounts, mixing ingredients, or scaling quantities)  
- Logical deduction using if-then reasoning or conditional statements with numerical implications  
- Coding/decoding (e.g., simple symbol-to-number substitution or pattern-based encoding)  
- Numerical analogies (e.g., “36 is to 6 as 81 is to ___”)  
- Unit conversion and measurement (e.g., converting between units of time, length, volume, weight, or currency)

Guidelines:  
- Ensure all problems are solvable without a calculator.  
- Use clear, concise language appropriate for the specified age group.  
- Avoid culturally biased or region-specific references (e.g., use generic currency symbols like “coins” or “units” unless conversion is purely numerical).  
- Maintain a balanced distribution across the listed skill areas (approximately 1-2 questions per category).

Output Format:  
- Begin with: Section Name: Mathematical Reasoning (10 questions, 10 minutes)  
- For each question, provide:  
  Question: [Full question text]  
  Options:  
  A) [Option text]  
  B) [Option text]  
  C) [Option text]  
  D) [Option text]  
- Answer Key:  
  1. [Correct Option]  
  2. [Correct Option]  
  ...  
  10. [Correct Option]

Example Output:

Section 2: Mathematical Reasoning (10 questions, 10 minutes)
1. Question: What number comes next in the sequence: 2, 4, 8, 16, __?
A) 24
B) 32
C) 20
D) 18

2. Question: If a pizza is divided into 8 equal slices and you eat 3 slices, what fraction of the pizza remains?
A) 3/8
B) 5/8
C) 1/2
D) 3/5

3. Question: A shirt costs $25 and is on sale for 20% off. What is the sale price?
A) $20
B) $22
C) $23
D) $24

4. Question: Complete the analogy: 36 is to 6 as 81 is to __?
A) 8
B) 9
C) 7
D) 10

5. Question: If a train travels 60 miles in 1.5 hours, what is its average speed in miles per hour?
A) 30 mph
B) 40 mph
C) 45 mph
D) 50 mph

6. Question: Solve: (15 ÷ 3) × 2 + 5 = ?
A) 10
B) 15
C) 20
D) 25

7. Question: If a code uses A=1, B=2, C=3, what is the value of CAT?
A) 24
B) 25
C) 26
D) 27

8. Question: How many minutes are in 3.5 hours?
A) 180
B) 200
C) 210
D) 220

9. Question: A recipe requires 2 cups of flour for 12 cookies. How many cups are needed for 30 cookies?
A) 4
B) 5
C) 6
D) 7

10. Question: What is the next number: 1, 1, 2, 3, 5, 8, __?
A) 11
B) 12
C) 13
D) 14

Answer Key:
1. B
2. B
3. A
4. B
5. B
6. B
7. A
8. C
9. B
10. C

"""
    }
    
    return prompts
