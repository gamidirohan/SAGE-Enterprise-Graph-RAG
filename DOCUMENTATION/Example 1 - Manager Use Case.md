## 🔹 Example 1: Manager Use Case

### Scenario

A project delivery manager asks:

> “Why was Project Orion delayed, and who was responsible for the final approval?”

### What a Traditional RAG Would Do

It retrieves some emails, maybe a project report, and generates:

> “Project Orion was delayed due to backend migration issues. Final approval was given by John Miller.”

Sounds confident. Looks fine.
But where did that come from? Which email? Which policy? Was that actually the *final* approval?

Now let’s see SAGE.

---

### What SAGE Would Return

**Answer:**
Project Orion was delayed due to backend migration dependency issues identified on March 3rd. Final approval was granted by John Miller on March 5th under Policy-IT-Deployment-2024.

**Provenance Bundle Includes:**

* Graph path:
  `Project:Orion → depends_on → Task:Backend Migration → approved_by → Person:John Miller`

* Document IDs:

  * Email_3245
  * Deployment_Report_March
  * Policy-IT-Deployment-2024.pdf

* Validated Cypher query used

* Timestamp trail

Now the manager can:

* Click the exact email
* See the dependency chain
* Confirm approval timing
* Verify policy reference

That’s operational clarity, not conversational fluff.



