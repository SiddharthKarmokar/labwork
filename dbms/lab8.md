# DBMS LAB
## Siddharth Karmokar
- 123CS0061
---

- ## Part A – First Normal Form (1NF)
This part explores the **atomicity rule of 1NF**, where multi-valued attributes (like a list of products in one column) are split into individual rows. Data redundancy is reduced, and querying becomes easier.

![er](../images/l8a1.png)
![er](../images/l8a2.png)
---

- ## Part B – Second Normal Form (2NF)
We examine a table that had a **partial dependency** where a non-key attribute (StudentName) depended only on part of the composite primary key (StudentID, CourseID). After decomposition into 2NF, we split student and enrollment info into separate tables.

![er](../images/l8b1.png)
![er](../images/l8b2.png)
![er](../images/l8b3.png)
![er](../images/l8b4.png)

---

- ## Part C – Third Normal Form (3NF)
This scenario demonstrates a **transitive dependency** (`EmpID → DeptID → DeptName`). To achieve 3NF, we created separate tables for employees and departments, ensuring that each non-key attribute only depends on the primary key.

![er](../images/l8c1.png)
![er](../images/l8c2.png)
![er](../images/l8c3.png)
![er](../images/l8c4.png)
![er](../images/l8c5.png)
![er](../images/l8c6.png)

- ## Part D – Boyce–Codd Normal Form (BCNF)
This part addresses a **mutual dependency** between `Professor` and `Room` (i.e., `Professor → Room` and `Room → Professor`), which violates BCNF. To fix it, we decomposed the original table into:
- A `PROFESSOR_ROOM` table (mapping professors to rooms),
- A `TEACHING_BCNF` table (storing subject assignments).

This ensures that **every determinant is a candidate key**, eliminating redundancy and anomalies.

![er](../images/l8d1.png)
![er](../images/l8d2.png)
![er](../images/l8d3.png)
![er](../images/l8d4.png)
![er](../images/l8d5.png)
![er](../images/l8d6.png)
---