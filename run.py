import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    TODO: replace with actual API call (OpenAI, etc.)

    For now, this function is a placeholder.
    It should return a string containing the model's response.
    """
    raise NotImplementedError("Implement call_llm with your chosen LLM provider.")


def write_log(event_type: str, payload: dict, logs_dir: Path):
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat()
    log_entry = {
        "timestamp": ts,
        "event_type": event_type,
        "payload": payload,
    }
    fname = logs_dir / f"{ts.replace(':', '-')}-{event_type}.json"
    with fname.open("w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2)



class DataAgent:
    def __init__(self, df: pd.DataFrame, logs_dir: Path):
        self.df = df
        self.logs_dir = logs_dir

    def summarize(self) -> dict:
        df = self.df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Basic aggregations
        by_date = (df
                   .groupby("date")
                   .agg({
                       "spend": "sum",
                       "impressions": "sum",
                       "clicks": "sum",
                       "purchases": "sum",
                       "revenue": "sum"
                   })
                   .reset_index())
        by_date["ctr"] = by_date["clicks"] / by_date["impressions"].replace(0, pd.NA)
        by_date["roas"] = by_date["revenue"] / by_date["spend"].replace(0, pd.NA)

        # ROAS by campaign over time
        by_campaign_date = (df
                            .groupby(["campaign_name", "date"])
                            .agg({
                                "spend": "sum",
                                "impressions": "sum",
                                "clicks": "sum",
                                "purchases": "sum",
                                "revenue": "sum"
                            })
                            .reset_index())
        by_campaign_date["ctr"] = by_campaign_date["clicks"] / by_campaign_date["impressions"].replace(0, pd.NA)
        by_campaign_date["roas"] = by_campaign_date["revenue"] / by_campaign_date["spend"].replace(0, pd.NA)

        # ROAS by audience type
        by_audience = (df
                       .groupby("audience_type")
                       .agg({
                           "spend": "sum",
                           "impressions": "sum",
                           "clicks": "sum",
                           "purchases": "sum",
                           "revenue": "sum"
                       })
                       .reset_index())
        by_audience["ctr"] = by_audience["clicks"] / by_audience["impressions"].replace(0, pd.NA)
        by_audience["roas"] = by_audience["revenue"] / by_audience["spend"].replace(0, pd.NA)

        # Low-CTR campaigns snapshot
        campaign_perf = (df
                         .groupby("campaign_name")
                         .agg({
                             "spend": "sum",
                             "impressions": "sum",
                             "clicks": "sum",
                             "purchases": "sum",
                             "revenue": "sum"
                         })
                         .reset_index())
        campaign_perf["ctr"] = campaign_perf["clicks"] / campaign_perf["impressions"].replace(0, pd.NA)
        campaign_perf["roas"] = campaign_perf["revenue"] / campaign_perf["spend"].replace(0, pd.NA)
        low_ctr = campaign_perf.nsmallest(10, "ctr")

        summary = {
            "overall_time_series": by_date.tail(60).to_dict(orient="records"),  # last 60 days
            "campaign_time_series_sample": by_campaign_date.groupby("campaign_name")
                                .apply(lambda g: g.tail(30).to_dict(orient="records"))
                                .to_dict(),
            "audience_summary": by_audience.to_dict(orient="records"),
            "low_ctr_campaigns": low_ctr.to_dict(orient="records"),
        }

        write_log("data_summary", {"summary_keys": list(summary.keys())}, self.logs_dir)
        return summary

    def get_low_ctr_creatives(self, threshold: float = 0.01, top_n: int = 20) -> pd.DataFrame:
        df = self.df.copy()
        df["ctr"] = df["clicks"] / df["impressions"].replace(0, pd.NA)
        low_ctr = df[df["ctr"] <= threshold].sort_values("ctr").head(top_n)
        write_log("low_ctr_selection", {"rows": len(low_ctr)}, self.logs_dir)
        return low_ctr



PLANNER_SYSTEM_PROMPT = """
You are the Planner Agent in a multi-agent marketing analytics system.

You receive:
- A user query about Facebook Ads performance.
- A compact JSON summary of performance over time and by campaign.

Your job:
1. Decompose the query into subtasks for downstream agents.
2. Focus on ROAS and CTR drivers.
3. Identify relevant time windows and segments (campaigns, audiences, countries, creative types).

Output strictly in JSON with the following schema:

{
  "primary_goal": "string",
  "subtasks": [
    {
      "id": "string",
      "description": "string",
      "target_metrics": ["roas", "ctr", "spend", "purchases", "revenue"],
      "time_window_hint": "string",
      "segments": ["campaign_name", "audience_type", "country", "creative_type"],
      "assigned_agent": "InsightAgent or CreativeAgent"
    }
  ],
  "notes_for_evaluator": "string"
}

Use a Think -> Analyze -> Plan reasoning structure internally, but only output JSON.
"""

def run_planner(user_query: str, data_summary: dict, logs_dir: Path) -> dict:
    user_prompt = json.dumps({
        "user_query": user_query,
        "data_summary": data_summary
    }, indent=2)

    response = call_llm(PLANNER_SYSTEM_PROMPT, user_prompt)
    write_log("planner_output_raw", {"response": response}, logs_dir)
    return json.loads(response)



INSIGHT_SYSTEM_PROMPT = """
You are the Insight Agent. You explain *why* performance changed.

Input:
- A single subtask from the Planner.
- Relevant data summaries (time series, campaign performance, audiences).

Your job:
1. Generate hypotheses explaining ROAS and CTR changes.
2. Consider factors like: audience fatigue, creative type performance, country mix, frequency, volume changes.

Use this reasoning structure explicitly in your own thinking:
- Think: outline possible causes.
- Analyze: relate causes to the numeric patterns in the summary.
- Conclude: produce clear hypotheses.

Return JSON with schema:

{
  "subtask_id": "string",
  "hypotheses": [
    {
      "id": "string",
      "description": "string",
      "expected_pattern": "string",
      "related_metrics": ["roas", "ctr", "spend", "purchases", "impressions"],
      "initial_confidence": 0.0 to 1.0
    }
  ]
}

Output only valid JSON.
"""

def run_insight(subtask: dict, data_summary: dict, logs_dir: Path) -> dict:
    user_prompt = json.dumps({
        "subtask": subtask,
        "data_summary": data_summary
    }, indent=2)

    response = call_llm(INSIGHT_SYSTEM_PROMPT, user_prompt)
    write_log("insight_output_raw", {"response": response, "subtask_id": subtask["id"]}, logs_dir)
    return json.loads(response)


EVALUATOR_SYSTEM_PROMPT = """
You are the Evaluator Agent. Your role is to *test* hypotheses using numeric evidence.

Input:
- A set of hypotheses from the Insight Agent.
- Aggregated metrics from the Data Agent (time series and segment summaries).
- Notes from the Planner, if any.

For each hypothesis:
1. Compare before vs after periods, or contrast segments (e.g., high vs low ROAS campaigns).
2. Extract concrete numeric evidence (percentage changes, absolute differences).
3. Adjust confidence based on how strongly the data supports the hypothesis.
4. If confidence < 0.4, attempt a brief reflection and consider revising the hypothesis wording.

Return JSON with schema:

{
  "validated_hypotheses": [
    {
      "id": "string",
      "subtask_id": "string",
      "description": "string",
      "evidence": [
        {
          "metric": "roas / ctr / spend / purchases / impressions / revenue",
          "detail": "string",
          "before_value": "number or null",
          "after_value": "number or null",
          "relative_change": "number or null"
        }
      ],
      "final_confidence": 0.0 to 1.0,
      "verdict": "supported | partially_supported | not_supported"
    }
  ]
}

Output only valid JSON.
"""

def run_evaluator(all_hypothesis_blocks: list, data_summary: dict,
                  planner_notes: str, logs_dir: Path) -> dict:
    user_prompt = json.dumps({
        "hypothesis_blocks": all_hypothesis_blocks,
        "data_summary": data_summary,
        "planner_notes": planner_notes
    }, indent=2)

    response = call_llm(EVALUATOR_SYSTEM_PROMPT, user_prompt)
    write_log("evaluator_output_raw", {"response": response}, logs_dir)
    return json.loads(response)


CREATIVE_SYSTEM_PROMPT = """
You are the Creative Improvement Generator.

Input:
- A table of low-CTR ads with fields:
  campaign_name, adset_name, ctr, creative_type, creative_message,
  audience_type, country, spend, impressions, clicks, purchases, roas.
- Optional examples of higher-performing creatives from the same dataset.

Your job:
1. Infer what kind of messaging and framing has worked historically.
2. For each low-CTR ad, propose multiple new variants.

Reasoning structure (implicit, don't print):
- Think: what is the core offer and audience?
- Analyze: why might this creative have low CTR (too generic, weak hook, unclear benefit)?
- Conclude: propose improved headlines, bodies, and CTAs.

Output JSON with schema:

{
  "recommendations": [
    {
      "campaign_name": "string",
      "adset_name": "string",
      "original_creative_message": "string",
      "original_ctr": "number",
      "audience_type": "string",
      "country": "string",
      "creative_type": "string",
      "suggested_creatives": [
        {
          "headline": "string",
          "primary_text": "string",
          "cta": "string",
          "rationale": "string"
        }
      ]
    }
  ]
}

All suggestions must be grounded in the tone and themes of the original creative_message.
Output only valid JSON.
"""

def run_creative_agent(low_ctr_df: pd.DataFrame, logs_dir: Path) -> dict:
    # Convert only needed columns to JSON-safe dict
    records = low_ctr_df[[
        "campaign_name", "adset_name", "ctr", "creative_type",
        "creative_message", "audience_type", "country",
        "spend", "impressions", "clicks", "purchases", "roas"
    ]].to_dict(orient="records")

    user_prompt = json.dumps({"low_ctr_ads": records}, indent=2)
    response = call_llm(CREATIVE_SYSTEM_PROMPT, user_prompt)
    write_log("creative_output_raw", {"response": response}, logs_dir)
    return json.loads(response)


def build_report_markdown(validated_insights: dict, creatives: dict, output_path: Path):
    """
    Create report.md summarizing everything for a marketer.
    """
    lines = []
    lines.append("# Facebook Performance Analysis Report\n")
    lines.append(f"*Generated at: {datetime.utcnow().isoformat()} UTC*\n")

    # High-level insight summary
    lines.append("## 1. Key Insights on ROAS and Performance\n")
    for h in validated_insights.get("validated_hypotheses", []):
        lines.append(f"### Hypothesis: {h['description']}\n")
        lines.append(f"- Verdict: **{h['verdict']}**")
        lines.append(f"- Confidence: **{h['final_confidence']:.2f}**\n")
        if h.get("evidence"):
            lines.append("**Evidence:**\n")
            for e in h["evidence"]:
                lines.append(
                    f"- {e['metric']}: {e['detail']} "
                    f"(before={e['before_value']}, after={e['after_value']}, Δ={e['relative_change']})"
                )
        lines.append("")

    # Creative recommendations
    lines.append("## 2. Creative Improvement Recommendations\n")
    for rec in creatives.get("recommendations", []):
        lines.append(f"### Campaign: {rec['campaign_name']} / {rec['adset_name']}\n")
        lines.append(f"- Original CTR: {rec['original_ctr']}")
        lines.append(f"- Audience: {rec['audience_type']} / Country: {rec['country']}")
        lines.append(f"- Original Creative:\n> {rec['original_creative_message']}\n")
        lines.append("**Suggested creatives:**\n")
        for idx, sug in enumerate(rec["suggested_creatives"], start=1):
            lines.append(f"- Variant {idx}:")
            lines.append(f"  - Headline: {sug['headline']}")
            lines.append(f"  - Primary Text: {sug['primary_text']}")
            lines.append(f"  - CTA: {sug['cta']}")
            lines.append(f"  - Rationale: {sug['rationale']}\n")

    # Action items
    lines.append("## 3. Recommended Next Actions\n")
    lines.append("- Pause or limit spend on clearly underperforming campaigns.")
    lines.append("- Test 2–3 of the highest-confidence creative recommendations first.")
    lines.append("- Monitor ROAS and CTR over 7–14 days post-change.\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Kasparro Agentic Facebook Performance Analyst")
    parser.add_argument("query", type=str, help="User query, e.g. 'Analyze ROAS drop'")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to FB Ads CSV")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    out_insights = outdir / "insights.json"
    out_creatives = outdir / "creatives.json"
    out_report = outdir / "report.md"

    # Load data
    df = pd.read_csv(args.data)
    data_agent = DataAgent(df, logs_dir)
    data_summary = data_agent.summarize()

    # Planner
    plan = run_planner(args.query, data_summary, logs_dir)

    # Insight for each InsightAgent subtask
    all_hypothesis_blocks = []
    for st in plan["subtasks"]:
        if st["assigned_agent"].lower().startswith("insight"):
            hyp_block = run_insight(st, data_summary, logs_dir)
            all_hypothesis_blocks.append(hyp_block)

    # Evaluator on all hypotheses
    validated_insights = run_evaluator(
        all_hypothesis_blocks,
        data_summary,
        plan.get("notes_for_evaluator", ""),
        logs_dir,
    )

    # Creative generation for low-CTR campaigns
    low_ctr_df = data_agent.get_low_ctr_creatives()
    creative_output = run_creative_agent(low_ctr_df, logs_dir)

    # Save JSON outputs
    out_insights.write_text(json.dumps(validated_insights, indent=2), encoding="utf-8")
    out_creatives.write_text(json.dumps(creative_output, indent=2), encoding="utf-8")

    # Build human-readable report
    build_report_markdown(validated_insights, creative_output, out_report)


if __name__ == "__main__":
    main()
