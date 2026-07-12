import json
from pathlib import Path

data = {
  "posts": [
    {"post": """Filebeat and other beats can directly send the message to ES but there are addition advantages while using logstash.

you can use filters in logstash based on inputs.
It support Grok filter and other filter plugins like csv , xml and many more.
It supports multiple codecs
you can use logstash as single point to control all pipelines
logstash can be monitored via Kibana GUI.
Logstash management is easy and can be performed via Kibana GUI.
Filebeat only support files as input but logstash support large array on inputs type"""}
  ]
}

out_path = Path("inputs") / "sample.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Wrote {out_path.resolve()}")
