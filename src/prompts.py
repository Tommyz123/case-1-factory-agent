"""
Prompt templates for the Factory Search Agent LLM interactions
"""

EXTRACTION_PROMPT = """You are an expert at extracting structured information about manufacturing facilities from search results.

TASK: Extract information about PHYSICAL MANUFACTURING FACILITIES for the company: {company}

CRITICAL RULES:
1. ONLY extract facilities that PRODUCE PHYSICAL GOODS (factories, plants, refineries, assembly plants, manufacturing sites)
2. ABSOLUTELY EXCLUDE:
   - Headquarters (HQ)
   - Corporate offices
   - Sales offices
   - Regional offices
   - Administrative buildings
   - R&D centers / Research centers / Development centers / Innovation centers
   - Distribution centers (unless they also manufacture)
   - Warehouses
   - Service centers
3. You MUST provide evidence_urls (the source URL where you found this information)
4. ONLY extract information that is EXPLICITLY stated in the search results - DO NOT make assumptions or hallucinate

SEARCH RESULTS:
{context}

OUTPUT FORMAT:
Return a JSON array of facilities. Each facility MUST have these fields:
{{
  "facilities": [
    {{
      "company_name": "exact company name",
      "facility_name": "name of the manufacturing facility",
      "facility_type": "type of facility (factory/plant/refinery/assembly/mill/foundry/etc.)",
      "full_address": "COMPLETE street address with street number, street name, city, state/province, postal code if available. GOOD examples: '1600 Amphitheatre Parkway, Mountain View, CA 94043, USA', '100 Queen St W, Toronto, ON M5H 2N2, Canada'. BAD examples: 'California', 'somewhere in Texas', 'Georgetown plant'",
      "country": "country name",
      "city": "city name",
      "latitude": null or float,
      "longitude": null or float,
      "evidence_urls": ["URL where this information was found - MANDATORY"],
      "confidence_score": 0.0 to 1.0
    }}
  ]
}}

IMPORTANT:
- If a facility name/address contains words like "headquarters", "HQ", "office", "R&D", "research" - DO NOT include it
- evidence_urls MUST be non-empty and MUST be from the provided search results
- If you're not sure if something is a manufacturing facility, DO NOT include it
- It's better to extract fewer high-quality results than many uncertain ones

Return ONLY the JSON, no additional text.
"""

VALIDATION_PROMPT = """You are a validation expert. Your task is to verify if a facility is a REAL MANUFACTURING/PRODUCTION FACILITY that produces physical goods or processes materials.

FACILITY TO VALIDATE:
- Name: {facility_name}
- Type: {facility_type}
- Address: {address}
- Evidence URL: {evidence}

CRITICAL VALIDATION RULES:
Answer "YES" if the facility does ANY of these:
1. Manufactures or assembles physical products (cars, electronics, machinery, etc.)
2. Processes raw materials into finished goods (refineries, chemical plants, mills)
3. Produces energy or refines petroleum products (power plants, refineries)
4. Processes food, beverages, or pharmaceuticals
5. Fabricates, casts, or forges materials (foundries, metalworks)

Answer "NO" ONLY if it clearly falls into these categories:
- Headquarters, corporate office, regional office, administrative building
- R&D center, research facility, innovation center, development center, laboratory
- Pure sales office, service center, training center
- Pure distribution center, warehouse, or logistics hub (with NO manufacturing)
- Retail store, showroom, or dealership

IMPORTANT GUIDANCE:
- For energy/oil/gas companies: refineries, processing plants, and production facilities are VALID (answer YES)
- If the facility name lacks obvious keywords but the type/address suggests production, lean towards YES
- If uncertain whether it's manufacturing or just distribution, check the facility_type field
- Only reject if there's clear evidence it's NOT a production facility

Based on the information provided, is this a valid manufacturing/production facility?

Answer with ONLY one word: YES or NO

Answer:"""

FACILITY_TYPE_CLASSIFIER_PROMPT = """Classify the type of manufacturing facility based on the description.

Facility Name: {facility_name}
Description: {description}

Choose the MOST SPECIFIC category that applies:
- Automotive Assembly Plant
- Electronics Factory
- Chemical Plant
- Oil Refinery
- Steel Mill
- Pharmaceutical Plant
- Food Processing Plant
- Textile Mill
- Semiconductor Fab
- Aircraft Assembly Plant
- General Manufacturing Plant
- Other (specify)

Return ONLY the category name, nothing else.

Category:"""

# Additional prompts for future enhancements

GEOCODING_PROMPT = """Extract or estimate the latitude and longitude for this address:

Address: {address}
City: {city}
Country: {country}

If you know the approximate coordinates, provide them. Otherwise, return null.

Return in JSON format:
{{
  "latitude": null or float between -90 and 90,
  "longitude": null or float between -180 and 180
}}

JSON:"""

EVIDENCE_QUALITY_PROMPT = """Rate the quality of this evidence URL for proving that a facility is a manufacturing site.

URL: {url}
Snippet: {snippet}

Rate from 0.0 (not credible) to 1.0 (highly credible) based on:
- Is it from an official company website? (+0.4)
- Is it from a reputable news source? (+0.3)
- Does it explicitly mention production/manufacturing? (+0.2)
- Is it recent (within 5 years)? (+0.1)

Return ONLY a number between 0.0 and 1.0.

Score:"""
