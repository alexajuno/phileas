from _engine import build_engine
eng, cfg = build_engine()
print("home:", cfg.home)
print("db_path:", cfg.db_path)
t = eng.start_thread(label="smoke")
tid = t["thread_id"]
from phileas.models import Event
ev = Event(text="Mara mentioned she works night shifts at the General.", thread_id=tid)
eng.save_event(ev)
res = eng.memorize(
    content="Mara works night shifts at Toronto General Hospital (the General).",
    memory_type="knowledge",
    source_event_id=ev.id,
    entities=[{"name": "Mara", "type": "Person"}, {"name": "Toronto General Hospital", "type": "Organization"}],
    detect_conflict=False,
)
print("memorized:", res.get("id"))
print("status:", eng.status())
