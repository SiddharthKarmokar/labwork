# DBMS LAB 5
## Siddharth Karmokar
- 123CS0061

# Employee-Department Management

## Part A: ER Diagram
![er1](../images/lab5_er1.jpg)  <!-- dummy ER diagram -->

- ## Part B
![er](../images/l5b1.png)
![er](../images/l5b2.png)
- ## Part C
![er](../images/l5c1.png)
![er](../images/l5c2.png)
![er](../images/l5c3.png)
![er](../images/l5c4.png)
![er](../images/l5c5.png)
![er](../images/l5c6.png)

- ## Part D
![er](../images/l5d1.png)
![er](../images/l5d2.png)
![er](../images/l5d3.png)
![er](../images/l5d4.png)
![er](../images/l5d5.png)
![er](../images/l5d6.png)

- ## Part E
![er](../images/l5e1.png)
![er](../images/l5e2.png)
![er](../images/l5e3.png)
![er](../images/l5e4.png)
![er](../images/l5e5.png)
![er](../images/l5e6.png)
![er](../images/l5e7.png)
![er](../images/l5e8.png)
![er](../images/l5e9.png)

- ## Part F
### **22. What happens if `faculty_user` tries to insert into `EMPLOYEES`?**

> If `faculty_user` **does not have the INSERT privilege** on the `EMPLOYEES` table, the operation will fail with a **permission error**:

```sql
ORA-01031: insufficient privileges
```

> To allow insertion, the following must be granted:

```sql
grant insert on employees to C##faculty_user;
```

---

### **23. Can `clerk_user` still update after you revoked the privilege?**

> **No**. Once the `UPDATE` privilege is revoked, `clerk_user` **can no longer perform update operations** on the table. Any attempt will result in a **permission denied error**:

```sql
ORA-01031: insufficient privileges
```

---

### **24. Why is `WITH GRANT OPTION` considered dangerous in real systems?**

> Because it allows a user to **grant the same privilege to others**, it can:

* Bypass controlled access,
* Lead to **unauthorized privilege escalation**,
* Make tracking and revoking permissions harder,
* Pose **security risks** in multi-user environments.

---

### **25. What is the effect of granting privileges to `PUBLIC`?**

> Granting to `PUBLIC` makes the privilege available to **all users in the database** — present and future.
> It’s **high-risk** because:

* Even unauthorized users gain access,
* It breaks the principle of least privilege,
* It may expose sensitive data or allow unintended actions.

