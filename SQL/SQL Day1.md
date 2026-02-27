# SQL Day1

1.选择部门30的所有员工

```sqlite
SELECT * FROM emp WHERE deptno = 30;
```

2.列出所有柜员（CLERK）的姓名、编号和部门编号。

```sqlite
`SELECT ename, empno, deptno FROM emp WHERE job = 'CLERK';
```

3.列出佣金高于薪金30%的员工姓名、职位和收入（薪金+佣金）;

```sqlite
SELECT ename, job, (sal + IFNULL(comm, 0)) AS income FROM emp WHERE comm > sal * 0.3;
```

4.找出部门30中所有经理和部门20中所有柜员的详细资料。

```sqlite
SELECT * FROM emp WHERE (deptno = 30 AND job = 'MANAGER') OR (deptno = 20 AND job = 'CLERK');
```



5.找出部门30中所有经理、部门20中所有柜员以及既不是经理又不是柜员并且薪金大于等于2000的员工的详细资料。

```sqlite
SELECT * FROM emp WHERE (deptno = 30 AND job = 'MANAGER') OR (deptno = 20 AND job = 'CLERK') OR (job NOT IN ('MANAGER', 'CLERK') AND sal >= 2000);
```

6.收取佣金的工作有哪些？

```sqlite
SELECT DISTINCT job FROM emp WHERE comm IS NOT NULL AND comm > 0;
```

7.找出不收取佣金或佣金低于100的员工。

```sqlite
SELECT * FROM emp WHERE comm IS NULL OR comm < 100;
```

8.显示不带有A的员工的姓名

```sqlite
SELECT ename FROM emp WHERE ename NOT LIKE '%A%';
```

9.显示姓名字段的任何位置包含E的所有员工的详细资料，显示的结果按照基本工资从高到低排序，如果基本工资相同则按照雇佣时间从早到晚排序，如果雇佣的日期相同，则按照职位的字母顺序排序。

```sqlite
SELECT * FROM emp WHERE ename LIKE '%E%' ORDER BY sal DESC, hiredate ASC, job ASC;
```

10.显示非经理员工的姓名、编号、部门编号和总收入INCOME（薪金+佣金），按照总收入从高到低排序；如果总收入相同，按照薪金从高到低排序；如果薪金相同，按照部门号从小到大排序。

```sqlite
SELECT ename, empno, deptno, (sal + IFNULL(comm, 0)) AS INCOME FROM emp WHERE job != 'MANAGER' ORDER BY INCOME DESC, sal DESC, deptno ASC;
```

