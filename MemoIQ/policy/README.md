# Policy Rules

## Overview

Policy rules are **optional** validation rules that check extracted field values against credit policy thresholds. If no policy rules file exists or it's empty, the system will still fill the template based on what's found in the reference documents.

## How It Works

1. **Template Filling**: The template is **always** filled based on information extracted from reference documents, regardless of policy rules.

2. **Policy Validation** (Optional): If `policy_rules.json` exists and contains rules, the system will:
   - Validate extracted fields against the rules
   - Generate validation warnings/errors
   - Include these in the validation report

3. **If No Policy Rules**: 
   - Template filling proceeds normally
   - No policy validation is performed
   - Only consistency checks are run

## Policy Rules File

The `policy_rules.json` file is a **manually created** file with example/default validation rules. It's not auto-generated.

### File Structure

```json
{
  "rules": [
    {
      "rule_id": "unique_rule_id",
      "field": "field_name_to_check",
      "condition": "max|min|range",
      "value": 5.0,  // For max/min
      "min_value": 0.0,  // For range
      "max_value": 1.0,  // For range
      "severity": "error|warning",
      "message": "Human-readable message"
    }
  ]
}
```

### Example Rules

The default `policy_rules.json` includes example rules for:
- Debt to EBITDA ratio (max threshold)
- Loan to value ratio (max threshold)
- Debt service coverage ratio (min threshold)
- Loan amount (must be positive)
- Interest rate (range validation)

## Creating Your Own Policy Rules

1. **Copy the default file**: `MemoIQ/policy/policy_rules.json`
2. **Edit the rules**: Add, modify, or remove rules as needed
3. **Field names**: Must match the `field_id` values from your template schema
4. **Conditions**:
   - `max`: Field value must be ≤ `value`
   - `min`: Field value must be ≥ `value`
   - `range`: Field value must be between `min_value` and `max_value`

## Disabling Policy Validation

To disable policy validation entirely:

1. **Option 1**: Delete or rename `policy_rules.json`
2. **Option 2**: Leave the file with an empty rules array: `{"rules": []}`

The system will log: "No policy rules found or rules list is empty. Skipping policy validation."

## Important Notes

- **Policy rules are optional** - Template filling works without them
- **Template filling is independent** - It happens regardless of policy validation
- **Validation is additive** - Policy checks add warnings/errors but don't block template filling
- **Reference documents are the source** - All field values come from reference documents, not policy rules

