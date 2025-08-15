import os
import json
from datetime import datetime, time
from typing import Optional, List

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlmodel import SQLModel, Field, Session, create_engine, select
from pydantic import BaseModel
from twilio.rest import Client
from dotenv import load_dotenv
import httpx

load_dotenv()

# --- Config ---
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER", "")
BASE_URL = os.getenv("PUBLIC_BASE_URL", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
RECORD_CALLS = os.getenv("RECORD_CALLS", "false").lower() == "true"

CALL_WIN_START = os.getenv("CALL_WINDOW_LOCAL_START", "08:00")
CALL_WIN_END = os.getenv("CALL_WINDOW_LOCAL_END", "20:00")

assert TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and BASE_URL, \
    "Missing Twilio or BASE_URL env vars."

client = Client(TWILIO_SID, TWILIO_TOKEN)

# --- DB Setup ---
engine = create_engine("sqlite:///data.db", echo=False)

class Contact(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    phone: str  # E.164 format recommended
    consent: bool = True
    dnc: bool = False
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CallLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    contact_id: Optional[int] = Field(default=None, foreign_key="contact.id")
    call_sid: Optional[str] = None
    status: str = "created"
    outcome: Optional[str] = None
    transcript: Optional[str] = None  # keep minimal; redact PHI in real usage
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    call_sid: str
    contact_id: Optional[int] = None
    history_json: str = "[]"
    updated_at: datetime = Field(default_factory=datetime.utcnow)

def init_db():
    SQLModel.metadata.create_all(engine)

# --- Templates ---
templates = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)

app = FastAPI(title="AI Call Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
def on_startup():
    init_db()

# --- Utility ---
def within_call_window(now_local: datetime) -> bool:
    try:
        h1, m1 = map(int, CALL_WIN_START.split(":"))
        h2, m2 = map(int, CALL_WIN_END.split(":"))
        start_t = time(hour=h1, minute=m1)
        end_t = time(hour=h2, minute=m2)
        return start_t <= now_local.time() <= end_t
    except Exception:
        return True  # fail open if misconfigured

def redact(text: str) -> str:
    # Minimal placeholder; replace with robust PHI/PII redactor
    return text.replace(TWILIO_FROM, "[redacted]")

async def llm_complete(history: List[dict]) -> str:
    """
    history: list of {"role": "system"|"user"|"assistant", "content": "<text>"}
    """
    if LLM_PROVIDER == "none":
        # Fallback: acknowledge and steer to completion
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"Got it. You said: {last_user}. Anything else I can help you with today?"
    try:
        if LLM_PROVIDER == "openai":
            # OpenAI-compatible Chat Completions
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": LLM_MODEL,
                "messages": history,
                "temperature": 0.3,
                "max_tokens": 200,
            }
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"].strip()

        elif LLM_PROVIDER == "anthropic":
            # Anthropic-compatible Messages API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            # Convert to Anthropic format: system + user/assistant pairs
            system_text = "\n".join(m["content"] for m in history if m["role"] == "system")
            conv = [m for m in history if m["role"] in ("user", "assistant")]
            messages = [{"role": m["role"], "content": m["content"]} for m in conv]
            payload = {
                "model": LLM_MODEL,
                "max_tokens": 200,
                "temperature": 0.3,
                "system": system_text,
                "messages": messages
            }
            async with httpx.AsyncClient(timeout=30) as http:
                r = await http.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                # Anthropic returns list of content blocks
                blocks = data["content"]
                text_out = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                return text_out or "Okay. What else can I help you with?"

        else:
            return "Thanks. I captured that. Anything else I can help you with?"
    except Exception:
        return "Understood. Anything else I can help you with before I wrap up?"

SYSTEM_PROMPT = (
    "You are a concise, friendly voice agent for a family medicine practice. "
    "Identify the practice as 'Family Medicine of the Rockies'. "
    "Primary tasks: confirm or reschedule appointments, capture brief reasons for visit, "
    "and mark do-not-call requests. Keep answers under 2 sentences. "
    "If someone asks to stop calls, reply 'Understood, I will add you to our do-not-call list.' "
    "Avoid discussing diagnoses or specific medical advice. "
)

# --- Admin UI ---
@app.get("/", response_class=HTMLResponse)
def index():
    with Session(engine) as s:
        contacts = s.exec(select(Contact).order_by(Contact.created_at.desc())).all()
        logs = s.exec(select(CallLog).order_by(CallLog.updated_at.desc()).limit(25)).all()
    tpl = templates.get_template("index.html")
    return tpl.render(contacts=contacts, logs=logs)

@app.post("/contacts/create")
def create_contact(
    name: str = Form(...),
    phone: str = Form(...),
    consent: str = Form("true"),
    notes: str = Form("")
):
    c = Contact(name=name.strip(), phone=phone.strip(), consent=(consent == "true"), notes=notes.strip() or None)
    with Session(engine) as s:
        s.add(c)
        s.commit()
    return RedirectResponse("/", status_code=303)

@app.post("/contacts/{contact_id}/toggle_dnc")
def toggle_dnc(contact_id: int):
    with Session(engine) as s:
        c = s.get(Contact, contact_id)
        if c:
            c.dnc = not c.dnc
            s.add(c)
            s.commit()
    return RedirectResponse("/", status_code=303)

# --- Call Orchestration ---
@app.post("/contacts/{contact_id}/call")
def start_call(contact_id: int):
    now = datetime.now()
    if not within_call_window(now):
        return Response("Outside call window.", status_code=400)

    with Session(engine) as s:
        c = s.get(Contact, contact_id)
        if not c:
            return Response("Contact not found.", status_code=404)
        if not c.consent or c.dnc:
            return Response("Contact is not eligible to call.", status_code=400)

        # Create log
        log = CallLog(contact_id=c.id, status="initiated")
        s.add(log); s.commit(); s.refresh(log)

    status_cb = f"{BASE_URL}/twilio/status?contact_id={contact_id}&log_id={log.id}"
    url = f"{BASE_URL}/twilio/voice?contact_id={contact_id}&log_id={log.id}"

    call = client.calls.create(
        to=c.phone,
        from_=TWILIO_FROM,
        url=url,
        status_callback=status_cb,
        status_callback_event=["initiated", "ringing", "answered", "completed"],
        record=RECORD_CALLS
    )

    # Persist Call SID
    with Session(engine) as s:
        log = s.get(CallLog, log.id)
        log.call_sid = call.sid
        s.add(log); s.commit()

    return RedirectResponse("/", status_code=303)

# --- Twilio Webhooks ---
@app.post("/twilio/status")
async def twilio_status(request: Request, contact_id: Optional[int] = None, log_id: Optional[int] = None):
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus", "unknown")
    with Session(engine) as s:
        log = s.get(CallLog, log_id) if log_id else None
        if log:
            log.status = call_status
            log.updated_at = datetime.utcnow()
            s.add(log); s.commit()
    return Response(status_code=200)

@app.post("/twilio/voice", response_class=Response)
async def twilio_voice(request: Request, contact_id: Optional[int] = None, log_id: Optional[int] = None):
    # Initial TwiML: introduce, disclosure, hand off to AI loop with speech Gather
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">
    Hello, this is Family Medicine of the Rockies calling with a brief automated assistant.
  </Say>
  <Pause length="1"/>
  <Say voice="alice">
    This call may be recorded for quality. If you prefer not to continue, you can hang up any time.
  </Say>
  <Redirect method="POST">{BASE_URL}/twilio/ai-loop?contact_id={contact_id}&log_id={log_id}&first=1</Redirect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.post("/twilio/ai-loop", response_class=Response)
async def ai_loop(request: Request, contact_id: Optional[int] = None, log_id: Optional[int] = None, first: Optional[str] = None):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    speech = (form.get("SpeechResult") or "").strip()
    user_hungup = form.get("CallStatus") == "completed"

    # Load conversation context
    with Session(engine) as s:
        convo = s.exec(select(Conversation).where(Conversation.call_sid == call_sid)).first()
        if not convo:
            convo = Conversation(call_sid=call_sid, contact_id=contact_id, history_json="[]")
            s.add(convo); s.commit(); s.refresh(convo)

        history: List[dict] = json.loads(convo.history_json)
        if not history:
            history.append({"role": "system", "content": SYSTEM_PROMPT})

        # First prompt
        if first == "1":
            assistant = (
                "I can help confirm or reschedule your appointment, capture a short reason for your visit, "
                "or add you to our do-not-call list. How can I help today?"
            )
        else:
            # Handle user input
            if speech:
                # Opt-out path
                if "stop" in speech.lower() or "do not call" in speech.lower():
                    # Mark DNC
                    c = s.get(Contact, contact_id) if contact_id else None
                    if c:
                        c.dnc = True
                        s.add(c); s.commit()
                    assistant = "Understood. I will add you to our do-not-call list. Thank you."
                    history.extend([{"role":"user","content": redact(speech)},
                                    {"role":"assistant","content": assistant}])
                    convo.history_json = json.dumps(history); convo.updated_at = datetime.utcnow()
                    s.add(convo); s.commit()
                    # End call
                    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">{assistant}</Say>
  <Hangup/>
</Response>"""
                    # Log outcome
                    log = s.get(CallLog, log_id) if log_id else None
                    if log:
                        log.outcome = "dnc_requested"
                        log.updated_at = datetime.utcnow()
                        s.add(log); s.commit()
                    return Response(content=twiml, media_type="application/xml")

                # Normal AI turn
                history.append({"role": "user", "content": redact(speech)})
                assistant = await llm_complete(history)
                history.append({"role": "assistant", "content": assistant})
            else:
                assistant = "Sorry, I didn’t catch that. How can I help?"

        # Persist conversation & transcript summary (minimal)
        convo.history_json = json.dumps(history)
        convo.updated_at = datetime.utcnow()
        s.add(convo); s.commit()

        # Update log
        log = s.get(CallLog, log_id) if log_id else None
        if log:
            log.transcript = (log.transcript or "")[-4000:] + f"\nU: {speech}\nA: {assistant}"
            log.updated_at = datetime.utcnow()
            s.add(log); s.commit()

    # Continue the loop with <Gather input="speech">
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="alice">{assistant}</Say>
  <Gather input="speech"
          action="{BASE_URL}/twilio/ai-loop?contact_id={contact_id}&log_id={log_id}"
          method="POST"
          language="en-US"
          speechTimeout="auto"
          timeout="5">
    <Say voice="alice">You can speak after the tone.</Say>
  </Gather>
  <Say voice="alice">Thanks for your time. Goodbye.</Say>
  <Hangup/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")
