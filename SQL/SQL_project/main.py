import sqlite3
import os
import hashlib
import time
from datetime import datetime
import sys


class FileCount:
    """ 自动化入库、智能打标、异常检测、高级检索"""

    def __init__(self, dataset_path, db_path='./db/file.db'):
        self.dataset_path = dataset_path
        self.db_path = db_path
        self.start_time = None
        self.total_files = 0
        self.processed_files = 0

        # 创建数据库目录
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 初始化数据库
        self.init_database()

    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建文件信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                create_time TEXT NOT NULL,
                file_path TEXT NOT NULL,
                md5_hash TEXT,
                tags TEXT,
                is_duplicate BOOLEAN DEFAULT FALSE,
                is_empty BOOLEAN DEFAULT FALSE,
                processed_time TEXT NOT NULL
            )
        ''')

        # 创建索引以提高查询性能
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON files(filename)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_extension ON files(extension)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_size ON files(size_bytes)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_create_time ON files(create_time)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON files(tags)')

        conn.commit()
        conn.close()

    def calculate_md5(self, file_path):
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"计算MD5失败 {file_path}: {e}")
            return ""

    def get_file_metadata(self, file_path):
        """获取文件的元数据"""
        try:
            file_name = os.path.basename(file_path)
            _, file_extension = os.path.splitext(file_name)
            file_size = os.path.getsize(file_path)
            create_timestamp = os.path.getctime(file_path)
            create_time = datetime.fromtimestamp(create_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            return {
                "filename": file_name,
                "extension": file_extension.lower(),
                "size_bytes": file_size,
                "create_time": create_time,
                "file_path": file_path,
                "md5_hash": self.calculate_md5(file_path) if file_size > 0 else "",
                "is_empty": file_size == 0
            }
        except Exception as e:
            print(f"获取文件元数据失败 {file_path}: {e}")
            return None

    def intelligent_tagging(self, file_info):
        """智能打标逻辑 - 简化版，只保留核心标签"""
        tags = []

        # 基于扩展名的标签（核心分类）
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        text_extensions = ['.txt', '.csv', '.xml', '.json']

        if file_info["extension"] in image_extensions:
            tags.append("图片文件")
        elif file_info["extension"] in text_extensions:
            tags.append("文本文件")

        return ",".join(tags) if tags else "未分类"

    def check_duplicates(self, file_info):
        """检查文件是否重复 - 仅依据MD5哈希值"""
        if not file_info["md5_hash"]:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM files 
            WHERE md5_hash = ?
        ''', (file_info["md5_hash"],))

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def update_progress(self):
        """更新进度显示"""
        self.processed_files += 1
        elapsed_time = time.time() - self.start_time
        progress = (self.processed_files / self.total_files) * 100

        # 避免除零错误
        if self.processed_files > 0:
            eta = (elapsed_time / self.processed_files) * (self.total_files - self.processed_files)
        else:
            eta = 0

        # 清空当前行并显示进度
        sys.stdout.write('\r')
        sys.stdout.write(f"进度: {self.processed_files}/{self.total_files} ({progress:.1f}%) | "
                         f"耗时: {elapsed_time:.1f}s | "
                         f"ETA: {eta:.1f}s")
        sys.stdout.flush()

    def process_files(self):
        """处理所有文件 """
        print("开始文件处理")
        self.start_time = time.time()

        # 清空数据库，避免重复数据
        self.clear_database()

        # 统计文件总数
        file_count = 0
        for root, dirs, files in os.walk(self.dataset_path):
            file_count += len(files)

        self.total_files = file_count
        self.processed_files = 0

        print(f"发现 {self.total_files} 个文件需要处理")

        # 遍历处理所有文件
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_path = os.path.join(root, file)

                # 获取文件元数据
                file_info = self.get_file_metadata(file_path)
                if not file_info:
                    continue

                # 智能打标
                file_info["tags"] = self.intelligent_tagging(file_info)

                # 检查重复
                file_info["is_duplicate"] = self.check_duplicates(file_info)

                # 存储到数据库
                self.store_to_database(file_info)

                # 更新进度
                self.update_progress()

        # 完成处理
        total_time = time.time() - self.start_time
        print(f"\n\n处理完成!")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"处理速度: {self.total_files / total_time:.2f} 文件/秒")

    def store_to_database(self, file_info):
        """存储文件信息到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO files 
            (filename, extension, size_bytes, create_time, file_path, md5_hash, tags, is_duplicate, is_empty, processed_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_info["filename"],
            file_info["extension"],
            file_info["size_bytes"],
            file_info["create_time"],
            file_info["file_path"],
            file_info["md5_hash"],
            file_info["tags"],
            file_info["is_duplicate"],
            file_info["is_empty"],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))

        conn.commit()
        conn.close()

    def clear_database(self):
        """清空数据库，避免重复数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM files') # 删除所有记录
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="files"')  # 重置自增ID计数器
        conn.commit()
        conn.close()
        print("已清空数据库，准备重新处理文件")

    def detect_anomalies(self):
        """异常数据检测"""
        print("\n开始异常数据检测")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. 查找空文件
        cursor.execute('SELECT filename, file_path FROM files WHERE is_empty = TRUE')
        empty_files = cursor.fetchall()

        print(f"\n1. 空文件 (0KB): {len(empty_files)} 个")
        for filename, file_path in empty_files[:10]:  # 只显示前10个
            print(f"   - {filename}: {file_path}")

        # 2. 查找重复文件
        cursor.execute('''
            SELECT md5_hash, COUNT(*) as count, GROUP_CONCAT(filename) as files
            FROM files 
            WHERE md5_hash != '' AND is_duplicate = TRUE
            GROUP BY md5_hash 
            HAVING count > 1
        ''')
        duplicate_groups = cursor.fetchall()

        print(f"\n2. 重复文件组: {len(duplicate_groups)} 组")
        for md5_hash, count, files in duplicate_groups[:5]:  # 只显示前5组
            file_list = files.split(',')
            print(f"   - MD5: {md5_hash[:8]}    重复数: {count}")
            for file in file_list[:3]:  # 只显示前3个文件名
                print(f"     * {file}")

        # 3. 简化标签统计
        # 只保留文本类和图片类统计
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN tags LIKE '%图片文件%' THEN '图片类'
                    WHEN tags LIKE '%文本文件%' THEN '文本类'
                    ELSE NULL
                END as category,
                COUNT(*) as count
            FROM files 
            WHERE tags != '未分类'
            GROUP BY category
            HAVING category IS NOT NULL
            ORDER BY count DESC
        ''')
        category_stats = cursor.fetchall()

        print(f"\n3. 主要标签分类统计:")
        for category, count in category_stats:
            print(f"   - {category}: {count} 个文件")

        conn.close()

    def advanced_queries(self):
        """高级检索功能"""
        print("\n=== 高级检索功能 ===")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        while True:
            print("\n请选择要执行的查询:")
            print("1. 查询名称中包含指定数字编号的图片数量")
            print("2. 查询占用空间最大的前N个文件")
            print("3. 查询特定年份创建的文本素材")
            print("4. 自定义SQL查询")
            print("0. 退出高级检索功能")

            choice = input("\n请输入选项(0-4): ").strip()

            if choice == '0':
                print("已退出高级检索功能")
                break

            elif choice == '1':
                nem = input("请输入要查询的数字: ").strip()
                if not nem.isdigit():
                    print("错误: 请输入有效的数字")
                    continue

                print(f"\n查询名称中包含'{nem}'的图片...")
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM files 
                    WHERE filename LIKE ? 
                    AND extension IN ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                ''', (f'%{nem}%',))

                count = cursor.fetchone()[0]
                print(f"结果: 共找到 {count} 张名称中包含'{nem}'的图片")

                if count > 0:
                    show_samples = input("是否显示示例文件?(y/n): ").strip().lower()
                    if show_samples == 'y':
                        cursor.execute('''
                            SELECT filename, file_path 
                            FROM files 
                            WHERE filename LIKE ? 
                            AND extension IN ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                            LIMIT 5
                        ''', (f'%{nem}%',))

                        print("\n示例文件:")
                        for filename, file_path in cursor.fetchall():
                            print(f"  - {filename}")

            elif choice == '2':
                try:
                    limit = int(input("请输入要显示的文件数量[默认为10]: ").strip() or "10")
                except ValueError:
                    print("错误: 请输入有效的数字")
                    continue

                print(f"\n查询占用空间最大的前{limit}个文件...")
                cursor.execute('''
                    SELECT filename, file_path, size_bytes 
                    FROM files 
                    ORDER BY size_bytes DESC 
                    LIMIT ?
                ''', (limit,))

                largest_files = cursor.fetchall()

                print(f"\n结果: 占用空间最大的前{limit}个文件:")
                for i, (filename, file_path, size_bytes) in enumerate(largest_files, 1):
                    size_mb = size_bytes / (1024 * 1024)
                    print(f"{i}. {filename}")
                    print(f"   大小: {size_mb:.2f} MB")
                    print(f"   路径: {file_path}")

            elif choice == '3':
                year = input("请输入年份: ").strip()
                if not year.isdigit():
                    print("错误: 请输入有效的年份")
                    continue

                print(f"\n查询{year}年创建的文本素材...")
                sql = "SELECT filename, create_time, file_path, size_bytes FROM files WHERE create_time LIKE ? AND extension IN ('.txt', '.csv', '.xml', '.json') ORDER BY create_time"
                cursor.execute(sql, (f'{year}-%',))

                old_text_files = cursor.fetchall()

                print(f"\n结果: 共找到 {len(old_text_files)} 个{year}年的文本素材")

                if old_text_files:
                    print("文件列表:")
                    for filename, create_time, file_path, size_bytes in old_text_files:
                        size_kb = size_bytes / 1024
                        print(f"- {filename}")
                        print(f"  创建时间: {create_time}")
                        print(f"  大小: {size_kb:.1f} KB")
                        print(f"  路径: {file_path}")
                else:
                    print("未找到符合条件的文件")

            elif choice == '4':
                print("\n自定义SQL查询")
                print("提示: 您可以直接输入SQL查询语句")
                sql_query = input("请输入SQL查询语句: ").strip()

                if not sql_query.lower().startswith('select'):
                    print("警告: 非SELECT查询可能修改数据库，请谨慎操作")
                    confirm = input("确认执行此查询?(y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                try:
                    cursor.execute(sql_query)
                    results = cursor.fetchall()

                    if not results:
                        print("查询结果为空")
                    else:
                        print(f"\n查询结果 (共{len(results)}条):")
                        for row in results:
                            print(row)

                except Exception as e:
                    print(f"查询执行错误: {e}")

            else:
                print("无效的选项，请重新输入")

        conn.close()


def main():
    """主函数"""
    dataset_path = './dataset/boss_files'
    db_path = 'db/file.db'

    # 创建文件计数系统实例
    etl_system = FileCount(dataset_path, db_path)

    # 执行ETL处理
    etl_system.process_files()

    # 异常检测
    etl_system.detect_anomalies()

    # 高级检索
    print("\n是否进入高级检索功能？(y/n):",end="")
    if input().strip().lower()=="y":
        etl_system.advanced_queries()

if __name__ == '__main__':
    main()