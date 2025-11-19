import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import threading
import pandas as pd
import duckdb

class DuckDBHandler:
    """
    DuckDB数据库操作处理类
    负责管理DuckDB关系数据库的操作
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化数据库连接
        """
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.config = config['db']
        self.db_path = Path(self.config['db_path']).resolve()
        self.sqlscript_path = Path(self.config['sqlscript_path']).resolve()
        os.makedirs(self.db_path.parent, exist_ok=True)

        # DuckDB初始化
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        # 添加线程锁，确保数据库操作的线程安全
        self._db_lock = threading.Lock()
        # 建立数据库连接
        self._connect()
        # 初始化数据库表
        self._init_tables()
    
    def _connect(self) -> bool:
        """
        建立数据库连接
        :return: 是否成功连接
        """
        try:
            # 使用DuckDB连接，支持并发读取
            self.conn = duckdb.connect(str(self.db_path))
            # 启用并行处理
            self.conn.execute("PRAGMA threads=4")
            # 启用内存映射
            self.conn.execute("PRAGMA enable_progress_bar=true")
            return True
        except duckdb.Error as e:
            self.logger.error(f"数据库连接错误: {e}")
            return False
    
    def _init_tables(self) -> None:
        """
        初始化数据库表
        从SQL脚本文件中读取并执行建表语句
        """
        try:
            with open(self.sqlscript_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            if self.conn:
                # 执行每个SQL语句单独执行，避免一次执行多个语句的问题
                for statement in sql_script.split(';'):
                    if statement.strip():
                        self.conn.execute(statement)
                self.logger.info("数据库表初始化成功")
            else:
                self.logger.error("数据库连接未建立")
        except duckdb.Error as e:
            self.logger.error(f"初始化表错误: {e}")
        except FileNotFoundError:
            self.logger.error(f"SQL脚本文件未找到: {self.sqlscript_path}")
        except Exception as e:
            self.logger.error(f"初始化表时发生未知错误: {e}")
    
    def close(self) -> None:
        """
        关闭数据库连接
        """
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def insert_detection_record(self, record: Dict[str, Any]) -> bool:
        """
        插入检测记录
        :param record: 检测记录字典
        :return: 是否插入成功
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return False
                
                # 构建插入语句
                columns = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record.keys()])
                sql = f"INSERT OR REPLACE INTO detection_history ({columns}) VALUES ({placeholders})"
                
                self.conn.execute(sql, list(record.values()))
                return True
        except duckdb.Error as e:
            self.logger.error(f"插入检测记录错误: {e}")
            return False
    
    def batch_insert_detection_records(self, records: List[Dict[str, Any]]) -> bool:
        """
        批量插入检测记录
        :param records: 检测记录列表
        :return: 是否插入成功
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return False
                
                # 转换为DataFrame，利用DuckDB的高效批量插入
                df = pd.DataFrame(records)

                # 使用DuckDB的DataFrame插入功能
                self.conn.table('detection_history').insert(df)
                return True
        except duckdb.Error as e:
            self.logger.error(f"批量插入检测记录错误: {e}")
            return False
    
    def query_detection_records(self, 
                               filters: Optional[Dict[str, Any]] = None, 
                               limit: Optional[int] = None,
                               offset: Optional[int] = None,
                               order_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        查询检测记录
        :param filters: 过滤条件
        :param limit: 限制返回记录数
        :param offset: 偏移量
        :param order_by: 排序字段
        :return: 检测记录列表
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return []
                
                # 构建查询语句
                sql = f"SELECT * FROM detection_history"
                
                # 添加过滤条件
                if filters:
                    where_conditions = []
                    for key, value in filters.items():
                        if isinstance(value, list):
                            # 处理IN查询
                            placeholders = ', '.join(['?' for _ in value])
                            where_conditions.append(f"{key} IN ({placeholders})")
                        elif isinstance(value, bool):
                            # 处理布尔值
                            where_conditions.append(f"{key} = {int(value)}")
                        else:
                            # 处理普通值
                            where_conditions.append(f"{key} = ?")
                    
                    if where_conditions:
                        sql += " WHERE " + " AND ".join(where_conditions)
                
                # 添加排序
                if order_by:
                    sql += f" ORDER BY {order_by}"
                
                # 添加分页
                if limit:
                    sql += f" LIMIT {limit}"
                    if offset:
                        sql += f" OFFSET {offset}"
                
                # 准备参数
                params = []
                if filters:
                    for key, value in filters.items():
                        if isinstance(value, list):
                            params.extend(value)
                        else:
                            params.append(value)
                
                # 执行查询
                result = self.conn.execute(sql, params).fetchall()
                
                # 转换为字典列表
                records = []
                for row in result:
                    records.append(dict(row))
                
                return records
        except duckdb.Error as e:
            self.logger.error(f"查询检测记录错误: {e}")
            return []
    
    def query_detection_records_by_date_range(self, 
                                            start_time: int, 
                                            end_time: int,
                                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        按时间范围查询检测记录
        :param start_time: 开始时间戳
        :param end_time: 结束时间戳
        :param filters: 额外的过滤条件
        :return: 检测记录列表
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return []
                
                # 构建查询语句
                sql = f"SELECT * FROM detection_history WHERE create_time BETWEEN ? AND ?"
                
                # 添加额外的过滤条件
                if filters:
                    for key, value in filters.items():
                        if isinstance(value, list):
                            # 处理IN查询
                            placeholders = ', '.join(['?' for _ in value])
                            sql += f" AND {key} IN ({placeholders})"
                        elif isinstance(value, bool):
                            # 处理布尔值
                            sql += f" AND {key} = {int(value)}"
                        else:
                            # 处理普通值
                            sql += f" AND {key} = ?"
                
                # 准备参数
                params = [start_time, end_time]
                if filters:
                    for key, value in filters.items():
                        if isinstance(value, list):
                            params.extend(value)
                        else:
                            params.append(value)
                
                # 执行查询
                result = self.conn.execute(sql, params).fetchall()
                
                # 转换为字典列表
                records = []
                for row in result:
                    records.append(dict(row))
                
                return records
        except duckdb.Error as e:
            self.logger.error(f"按时间范围查询检测记录错误: {e}")
            return []
    
    def get_detection_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取检测记录
        :param record_id: 记录ID
        :return: 检测记录
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return None
                
                # 构建查询语句
                sql = f"SELECT * FROM detection_history WHERE id = ?"
                
                # 执行查询
                result = self.conn.execute(sql, [record_id]).fetchone()
                
                if result:
                    return dict(result)
                
                return None
        except duckdb.Error as e:
            self.logger.error(f"根据ID获取检测记录错误: {e}")
            return None
    
    def update_detection_record(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新检测记录
        :param record_id: 记录ID
        :param updates: 更新字段
        :return: 是否更新成功
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return False
                
                # 构建更新语句
                set_clauses = []
                for key in updates.keys():
                    set_clauses.append(f"{key} = ?")
                
                sql = f"UPDATE detection_history SET {', '.join(set_clauses)} WHERE id = ?"
                
                # 准备参数
                params = list(updates.values())
                params.append(record_id)
                
                # 执行更新
                self.conn.execute(sql, params)
                return True
        except duckdb.Error as e:
            self.logger.error(f"更新检测记录错误: {e}")
            return False
    
    def delete_detection_record(self, record_id: str) -> bool:
        """
        删除检测记录
        :param record_id: 记录ID
        :return: 是否删除成功
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return False
                
                # 执行删除
                self.conn.execute("DELETE FROM detection_history WHERE id = ?", [record_id])
                return True
        except duckdb.Error as e:
            self.logger.error(f"删除检测记录错误: {e}")
            return False
    
    def get_detection_statistics(self, 
                               filters: Optional[Dict[str, Any]] = None,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None) -> Dict[str, Any]:
        """
        获取检测统计信息
        :param filters: 过滤条件
        :param start_time: 开始时间戳
        :param end_time: 结束时间戳
        :return: 统计信息字典
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return {}
                
                # 构建基础查询
                base_sql = "FROM detection_history"
                where_conditions = []
                params = []
                
                # 添加时间范围条件
                if start_time is not None and end_time is not None:
                    where_conditions.append("create_time BETWEEN ? AND ?")
                    params.extend([start_time, end_time])
                
                # 添加过滤条件
                if filters:
                    for key, value in filters.items():
                        if isinstance(value, list):
                            # 处理IN查询
                            placeholders = ', '.join(['?' for _ in value])
                            where_conditions.append(f"{key} IN ({placeholders})")
                            params.extend(value)
                        elif isinstance(value, bool):
                            # 处理布尔值
                            where_conditions.append(f"{key} = {int(value)}")
                        else:
                            # 处理普通值
                            where_conditions.append(f"{key} = ?")
                            params.append(value)
                
                if where_conditions:
                    base_sql += " WHERE " + " AND ".join(where_conditions)
                
                # 执行统计查询
                total_count = self.conn.execute(f"SELECT COUNT(*) {base_sql}", params).fetchone()
                total_count = total_count[0] if total_count else 0
                illegal_count = self.conn.execute(f"SELECT COUNT(*) {base_sql} AND is_illegal = 1", params).fetchone()
                illegal_count = illegal_count[0] if illegal_count else 0
                aigc_count = self.conn.execute(f"SELECT COUNT(*) {base_sql} AND is_aigc = 1", params).fetchone()
                aigc_count = aigc_count[0] if aigc_count else 0
                
                # 按标签统计
                label_stats = self.conn.execute(f"""
                    SELECT labels, COUNT(*) as count 
                    {base_sql} AND labels IS NOT NULL AND labels != ''
                    GROUP BY labels
                    ORDER BY count DESC
                    LIMIT 10
                """, params).fetchall()
                
                # 按分级统计
                grading_stats = self.conn.execute(f"""
                    SELECT grading, COUNT(*) as count 
                    {base_sql} AND grading IS NOT NULL AND grading != ''
                    GROUP BY grading
                    ORDER BY count DESC
                """, params).fetchall()
                
                # 按分类统计
                classification_stats = self.conn.execute(f"""
                    SELECT classification, COUNT(*) as count 
                    {base_sql} AND classification IS NOT NULL AND classification != ''
                    GROUP BY classification
                    ORDER BY count DESC
                    LIMIT 10
                """, params).fetchall()
                
                # 按平台统计
                platform_stats = self.conn.execute(f"""
                    SELECT publish_platform, COUNT(*) as count 
                    {base_sql} AND publish_platform IS NOT NULL AND publish_platform != ''
                    GROUP BY publish_platform
                    ORDER BY count DESC
                    LIMIT 10
                """, params).fetchall()
                
                # 返回统计结果
                return {
                    "total_count": total_count,
                    "illegal_count": illegal_count,
                    "aigc_count": aigc_count,
                    "illegal_rate": round(illegal_count / total_count * 100, 2) if total_count > 0 else 0,
                    "aigc_rate": round(aigc_count / total_count * 100, 2) if total_count > 0 else 0,
                    "label_stats": [dict(row) for row in label_stats],
                    "grading_stats": [dict(row) for row in grading_stats],
                    "classification_stats": [dict(row) for row in classification_stats],
                    "platform_stats": [dict(row) for row in platform_stats]
                }
        except duckdb.Error as e:
            self.logger.error(f"获取检测统计信息错误: {e}")
            return {}
    
    def export_detection_records(self, 
                               filters: Optional[Dict[str, Any]] = None,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None,
                               format: str = "parquet",
                               output_path: Optional[str] = None) -> str:
        """
        导出检测记录
        :param filters: 过滤条件
        :param start_time: 开始时间戳
        :param end_time: 结束时间戳
        :param format: 导出格式，支持parquet、csv、json
        :param output_path: 输出路径，如果为None则自动生成
        :return: 导出文件路径
        """
        try:
            with self._db_lock:
                if not self.conn:
                    self.logger.error("数据库连接未建立")
                    return ""
                
                # 构建查询语句
                sql = "SELECT * FROM detection_history"
                
                where_conditions = []
                params = []
                
                # 添加时间范围条件
                if start_time is not None and end_time is not None:
                    where_conditions.append("create_time BETWEEN ? AND ?")
                    params.extend([start_time, end_time])
                
                # 添加过滤条件
                if filters:
                    for key, value in filters.items():
                        if isinstance(value, list):
                            # 处理IN查询
                            placeholders = ', '.join(['?' for _ in value])
                            where_conditions.append(f"{key} IN ({placeholders})")
                            params.extend(value)
                        elif isinstance(value, bool):
                            # 处理布尔值
                            where_conditions.append(f"{key} = {int(value)}")
                        else:
                            # 处理普通值
                            where_conditions.append(f"{key} = ?")
                            params.append(value)
                
                if where_conditions:
                    sql += " WHERE " + " AND ".join(where_conditions)
                
                # 执行查询并获取结果
                result = self.conn.execute(sql, params).arrow()
                
                # 生成输出路径
                if not output_path:
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path(self.db_path).parent / "exports"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    if format.lower() == "parquet":
                        output_path = str(output_dir / f"detection_records_{timestamp}.parquet")
                    elif format.lower() == "csv":
                        output_path = str(output_dir / f"detection_records_{timestamp}.csv")
                    elif format.lower() == "json":
                        output_path = str(output_dir / f"detection_records_{timestamp}.json")
                    else:
                        raise ValueError(f"不支持的导出格式: {format}")
                
                # 根据格式导出
                if format.lower() == "parquet":
                    result.to_pandas().to_parquet(output_path, compression='zstd', index=False)
                elif format.lower() == "csv":
                    result.to_pandas().to_csv(output_path, index=False)
                elif format.lower() == "json":
                    result.to_pandas().to_json(output_path, orient='records', force_ascii=False, indent=2)
                
                self.logger.info(f"成功导出检测记录到: {output_path}")
                return output_path
        except Exception as e:
            self.logger.error(f"导出检测记录错误: {e}")
            return ""