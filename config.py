# Configuration settings for the application

# Paths for document directories
DEAL_DETAILS_DIR = "./data/Deal Details"
CASE_STUDIES_DIR = "./data/Case Studies"
SALES_PITCH_DIR = "./data/Sales Pitch"

# Document loader settings
GLOB_PATTERN = '**/*.json'
SHOW_PROGRESS = True

# Text splitter settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Model selection
LLM_MODEL = "llama3"
