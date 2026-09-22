import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from supabase import create_client, Client

app = FastAPI()

# Enable CORS for Vercel Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Data Schemas
class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total_price: float

class InvoiceExtraction(BaseModel):
    vendor_name: str
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    grand_total: float
    calculation_mismatch: bool
    line_items: List[LineItem]

@app.post("/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")

        # 1. Call OpenAI Vision API
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Extract structured data from this invoice. Set calculation_mismatch to true if total does not equal sum of items."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract invoice details:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            response_format=InvoiceExtraction
        )
        
        data = response.choices[0].message.parsed

        # 2. Save directly into Supabase DB
        db_payload = {
            "vendor_name": data.vendor_name,
            "invoice_number": data.invoice_number,
            "invoice_date": data.invoice_date if data.invoice_date else None,
            "grand_total": data.grand_total,
            "has_mismatch": data.calculation_mismatch,
            "status": "needs_review" if data.calculation_mismatch else "approved",
            "raw_json": data.model_dump()
        }
        
        insert_res = supabase.table("invoices").insert(db_payload).execute()
        
        return {"success": True, "record": insert_res.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import os
import uvicorn

# (Keep all your existing FastAPI / OpenAI code above this line)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
