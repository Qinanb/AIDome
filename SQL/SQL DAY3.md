# SQL DAY3

1.列出至少有四个员工的所有部门编号、部门名称、部门人数

 ```sqlite
 SELECT dept.deptno,dname,COUNT(empno) 
 FROM dept LEFT JOIN emp ON dept.deptno=emp.deptno 
 GROUP BY dept.deptno 
 HAVING COUNT(empno)>=4;
 ```



2.列出雇佣日期早于其领导的所有员工的编号，姓名，部门名称，领导名称。

 ```sqlite
 SELECT tem.empno '员工编号',tem.en'员工姓名',d.dname '部门名称',tem.bn '领导名称'
 FROM dept d LEFT JOIN (
 	SELECT e.empno,e.deptno dn,e.ename en,m.ename bn
 	FROM emp e LEFT JOIN emp m ON e.mgr=m.empno
 	WHERE e.hiredate < m.hiredate
 	)tem
 WHERE tem.dn=d.deptno;
 ```



3.列出所有部门名称和这些部门的员工信息（人数、平均工资、平均服务年限、最高最低工资），同时列出那些没有员工的部门。

 ```sqlite
 SELECT d.deptno, dname ,COUNT(empno),AVG(sal),AVG((JULIANDAY('now')-JULIANDAY(hiredate))/365),MAX(sal),MIN(sal)
 FROM dept d LEFT JOIN emp e ON d.deptno=e.deptno
 GROUP BY d.deptno; 
 ```



4.列出最低工资大于1500的职位名称和以及此职位的人数。

 ```sqlite
 SELECT job,COUNT(empno)
 FROM emp
 GROUP BY job
 HAVING MIN(sal)>1500;
 ```



5.列出各个部门的CLERK（柜员）的最低薪金。

 ```sqlite
 SELECT deptno ,MIN(sal)
 FROM emp
 WHERE job='CLERK'
 GROUP BY deptno;
 ```



6.求出部门名称中带‘S’字符的部门员工的工资合计和部门的人数

```sqlite
SELECT d.deptno,dname,COUNT(sal),COUNT(empno)
FROM dept d LEFT JOIN emp e ON d.deptno=e.deptno
WHERE dname LIKE'%S%'
GROUP BY d.deptno;
```

