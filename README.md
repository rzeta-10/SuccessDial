# Predict the Success of Bank telemarketing

## Project Details
**Project**: Completed as part of MLP Project - IITM BS Degree  

**Accuracy**:  
- **0.72459**: Calculated with approximately 59% of the test data.  
- **0.73550**: Calculated with approximately 41% of the test data.  

---

## Dataset Description

The dataset contains data related to direct marketing campaigns of a banking institution. The marketing campaigns were conducted via phone calls. Often, multiple contacts with the same client were required to determine if the client would subscribe to the bank's term deposit.

### Files
- **`train.csv`**: The training dataset.
- **`test.csv`**: The test dataset.
- **`sample_submission.csv`**: A sample submission file in the correct format.

---

### Input Variables

1. **`last contact date`**: The date of the last contact with the client.  
2. **`age`** (numeric): The age of the client.  
3. **`job`**: The type of job the client has (categorical).  
4. **`marital`**: Marital status (categorical: `married`, `divorced`, `single`; note: "divorced" includes divorced and widowed).  
5. **`education`** (categorical): Level of education (`unknown`, `secondary`, `primary`, `tertiary`).  
6. **`default`**: Does the client have credit in default? (binary: `yes`, `no`).  
7. **`balance`**: The average yearly balance of the client, in euros (numeric).  
8. **`housing`**: Does the client have a housing loan? (binary: `yes`, `no`).  
9. **`loan`**: Does the client have a personal loan? (binary: `yes`, `no`).  
10. **`contact`**: Contact communication type (categorical: `unknown`, `telephone`, `cellular`).  
11. **`duration`**: Duration of the last contact, in seconds (numeric).  
12. **`campaign`**: Number of contacts performed during this campaign for the client (numeric, includes the last contact).  
13. **`pdays`**: Number of days since the client was last contacted in a previous campaign (numeric; `-1` means the client was not previously contacted).  
14. **`previous`**: Number of contacts performed before this campaign for the client (numeric).  
15. **`poutcome`**: Outcome of the previous marketing campaign (categorical: `unknown`, `other`, `failure`, `success`).  

---

### Output Variable (Target)

16. **`target`**: Has the client subscribed to a term deposit? (binary: `yes`, `no`).  
