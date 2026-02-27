# SQL Day2

1.查询公司工资等级在3以上（包含3），的雇员编号和姓名

 ```sqlite
 SELECT empno,ename FROM emp LEFT JOIN salgrade ON sal BETWEEN losal AND hisal WHERE grade>=3;
 ```



2.查询公司所有雇员的姓名，雇员编号，部门名称，要求部门名称小写且显示所有部门

 ```sqlite
 SELECT ename,empno,LOWER(dname) FROM dept LEFT JOIN emp ON emp.deptno=dept.deptno;
 ```



3.要求算出部门30的所有人员的姓名、雇佣年份（要求四舍五入到一位小数）、部门位置。

 ```sqlite
 SELECT ename,ROUND(STRFTIME('%Y',hiredate)+(STRFTIME('%m', hiredate)-1)/12.0,1 ),loc 
 FROM emp LEFT JOIN dept ON emp.deptno=dept.deptno WHERE emp.deptno=30;
 ```



4.查询所有在12月雇佣的雇员姓名、雇员编号、年薪（12*（薪金+佣金））、薪资等级

 ```sqlite
 SELECT ename,empno,12*(sal+IFNULL(comm,0)) '年薪',grade 
 FROM emp LEFT JOIN salgrade ON sal BETWEEN losal AND hisal WHERE STRFTIME('%m',hiredate)='12';
 ```



5.查询出所有雇员和其领导的姓名和年薪。

 ```sqlite
 SELECT e.ename '雇员姓名',12*(e.sal+IFNULL(e.comm,0)) '员工年薪',m.ename '领导姓名',12*(m.sal+IFNULL(m.comm,0)) '领导年薪' 
 FROM emp e LEFT JOIN emp m ON m.empno=e.mgr; 
 ```



6.要求显示员工的姓名，编号和部门名称（要求部门名称简写为首字母，员工姓名为小写）

```sqlite
SELECT LOWER(ename),empno,SUBSTR(dname,1,1) FROM emp LEFT JOIN dept ON emp.deptno=dept.deptno;
```

