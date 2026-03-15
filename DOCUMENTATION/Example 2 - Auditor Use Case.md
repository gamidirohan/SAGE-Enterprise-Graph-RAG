
## 🔹 Example 2: Auditor Use Case

Now let’s raise the stakes.

### Scenario

An internal auditor asks:

> “List all procurement approvals above ₹10 lakhs in Q3 2025 that violated escalation policy.”

This is multi-hop, policy-constrained, temporal reasoning.

Traditional RAG?
It would likely summarize loosely and miss violations.

---

### What SAGE Does

Step 1: GAP planner decomposes:

* Identify procurement transactions
* Filter by amount > ₹10L
* Filter by Q3 2025
* Check escalation requirement
* Validate approval chain

Step 2: Graph traversal finds:

* Transaction nodes
* Associated approval nodes
* Escalation requirement edges

Step 3: G-CT verifies the reasoning path exists in Neo4j.

Step 4: Policy_guard checks escalation compliance.

---

### Final Output

**Violations Found:**

1. Transaction ID: PROC_7782

   * Amount: ₹14.2L
   * Approved by: Regional Manager
   * Missing escalation to Finance Director
   * Violates Policy-FIN-ESC-2025

2. Transaction ID: PROC_7819

   * Amount: ₹22.8L
   * Escalation occurred after approval (non-compliant order)

**Attached:**

* Graph paths for each violation
* Cypher queries executed
* Document IDs
* Policy node references
* Timestamps

Now the auditor has:

* Evidence chain
* Deterministic reasoning
* Compliance traceability
* No hallucinated relationships

That’s legally defensible.

---

## 🔹 What Makes This Powerful

Managers use it to:

* Understand responsibility chains
* Analyze dependencies
* Validate decisions quickly

Auditors use it to:

* Detect violations
* Prove non-compliance
* Reconstruct decision trails
* Validate that reasoning paths exist in the data

The difference is subtle but profound:

Traditional RAG answers questions.

SAGE proves answers.

And in enterprise systems, proof beats fluency every single time.