"""
Main entry point for Factory Search Agent
"""
import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.graph import run_factory_search
from src.models import CompanyInput

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('factory_search.log')
    ]
)

logger = logging.getLogger(__name__)


def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()

    # Verify required environment variables
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please create a .env file based on .env.example")
        sys.exit(1)

    logger.info("Environment variables loaded successfully")
    logger.info(f"Using Azure OpenAI deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
    logger.info(f"Using Azure OpenAI embedding: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}")
    logger.info(f"Search method: {os.getenv('SEARCH_METHOD', 'web_scraping')}")


def main():
    """Main function"""
    logger.info("=" * 80)
    logger.info("Factory Search Agent - Starting")
    logger.info("=" * 80)

    # Load environment
    load_environment()

    # Read input file
    input_file = Path("examples/input_example.json")
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # Validate input
    try:
        company_input = CompanyInput(**input_data)
        companies = company_input.companies
    except Exception as e:
        logger.error(f"Invalid input format: {e}")
        sys.exit(1)

    logger.info(f"Processing {len(companies)} companies: {companies}")

    # Process companies in parallel
    all_facilities = []
    all_metadata = []

    # Helper function for parallel processing
    def process_single_company(company: str, idx: int) -> tuple:
        """Process a single company"""
        logger.info("")
        logger.info(f"[{idx}/{len(companies)}] Processing company: {company}")
        logger.info("-" * 80)

        try:
            result = run_factory_search(company)
            logger.info(f"[OK] Found {len(result.facilities)} facilities for {company}")
            return result.facilities, result.metadata, None
        except Exception as e:
            logger.error(f"[X] Error processing {company}: {e}", exc_info=True)
            return [], {"company": company, "error": str(e)}, str(e)

    # Parallel processing with ThreadPoolExecutor
    max_workers = min(len(companies), 3)  # Max 3 concurrent companies
    logger.info(f"Using {max_workers} parallel workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_company = {
            executor.submit(process_single_company, company, idx): company
            for idx, company in enumerate(companies, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_company):
            company = future_to_company[future]
            try:
                facilities, metadata, error = future.result()
                all_facilities.extend(facilities)
                all_metadata.append(metadata)

                if error:
                    logger.error(f"[FAIL] {company}: {error}")
            except Exception as e:
                logger.error(f"[FAIL] {company}: Unexpected error: {e}")
                all_metadata.append({"company": company, "error": str(e)})

    # Create final output
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"Total facilities found: {len(all_facilities)}")
    logger.info("=" * 80)

    # Convert to JSON-serializable format
    output_data = {
        "facilities": [facility.model_dump() for facility in all_facilities],
        "metadata": {
            "total_facilities": len(all_facilities),
            "companies_processed": companies,
            "company_details": all_metadata
        }
    }

    # Write output file
    output_file = Path("examples/output_example.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"[OK] Results written to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Companies processed: {len(companies)}")
    print(f"Total facilities found: {len(all_facilities)}")
    print(f"Output file: {output_file}")
    print("=" * 80)

    # Critical validation checks
    print("\n" + "!" * 80)
    print("CRITICAL VALIDATION CHECKS")
    print("!" * 80)

    # Check 1: Evidence URLs
    facilities_without_evidence = [
        f for f in all_facilities
        if not f.evidence_urls or len(f.evidence_urls) == 0
    ]
    if facilities_without_evidence:
        print(f"[!] WARNING: {len(facilities_without_evidence)} facilities missing evidence_urls")
        print("   This violates the automatic fail condition!")
    else:
        print(f"[OK] All {len(all_facilities)} facilities have evidence_urls")

    # Check 2: Keyword check for HQ/R&D/Office
    exclude_keywords = ['headquarters', 'hq', 'office', 'r&d', 'research', 'development']
    facilities_with_excluded = []

    for facility in all_facilities:
        text = f"{facility.facility_name} {facility.full_address}".lower()
        for keyword in exclude_keywords:
            if keyword in text:
                facilities_with_excluded.append((facility.facility_name, keyword))
                break

    if facilities_with_excluded:
        print(f"[!] WARNING: {len(facilities_with_excluded)} facilities contain excluded keywords:")
        for name, keyword in facilities_with_excluded[:5]:
            print(f"   - {name} (contains '{keyword}')")
        print("   This violates the automatic fail condition!")
    else:
        print(f"[OK] No facilities contain HQ/R&D/Office keywords")

    print("!" * 80)

    logger.info("Factory Search Agent - Completed")


if __name__ == "__main__":
    main()
