import json
import logging
from hashlib import sha256
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from src.db_handler import DuckDBHandler

class HistoryManager:
    """
    检测历史记录管理类
    """
    def __init__(self, config: dict, db_handler: DuckDBHandler):
        self.logger = logging.getLogger(f"IHI_detection.{__name__}")
        self.config = config['history']
        self.db_handler = db_handler
        # 定义需要进行JSON转换的字段名
        self.json_fields = ['metadata', 'analytical_notes']
    
    def get_urlid(self, url: str) -> str:
        """
        获取URL计算的ID
        :param url: 要计算的URL
        :return: ID，失败返回空字符串
        """
        if "xsec_token" in url:
            return sha256(url.split("?")[0].encode()).hexdigest()
        else:
            return sha256(url.encode()).hexdigest()

    def insert_detection_history(self,
        content: str,
        disposal_status: int,
        source_url: str = '',
        presentation_form: str = '文本',
        publish_platform: str = '',
        publisher_id: str = '',
        publish_time: int = 0,
        ip_region: str = '',
        labels: str = "",
        grading: str = "",
        classification: str = "",
        is_illegal: bool = False,
        is_aigc: bool = False,
        rumor_detection_conclusion: str = '',
        harmlevel: str = '',
        reasoning: str = '',
        suggestions: str = '',
        law_ref: str = '',
        attachment_count: int = 0,
        ref_urls: str = '',
        **metadata) -> bool:
        """
        插入或忽略检测历史记录
        
        Args:
            content: 被检测的原始内容
            disposal_status: 处置状态，0无须处置，1未推送处置，2已推送处置，3已处置完毕
            source_url: 内容来源URL，可选
            presentation_form: 内容呈现形式，文本/图文/视频，可选
            publish_platform: 内容发布平台，可选
            publisher_id: 内容发布平台的用户ID，可选
            publish_time: 内容发布秒级时间戳，可选
            ip_region: 内容来源IP所属区域，可选
            labels: 检测结果标签，英文逗号分隔，可选
            grading: 检测结果等级，可选
            classification: 检测结果分类，可选
            is_illegal: 被检测的原始内容是否为违法信息，可选
            is_aigc: 被检测的原始内容是否为AI生成内容，可选
            rumor_detection_conclusion: 谣言检测结论，可选
            harmlevel: 危害等级，可选
            reasoning: 推理过程，可选
            suggestions: 建议，可选
            law_ref: 法律参考，可选
            attachment_count: 附件的数量，可选
            ref_urls: 参考信息的链接列表，英文逗号分隔，可选
            metadata: 其他自定义参数
            
        Raises:
            Exception: 当数据库连接异常时抛出
        """
        # 参数验证
        content = content.strip()
        if not content:
            self.logger.error("content不能为空")
            return False
        else:
            # 截断过长的内容以防止数据库错误
            content = content[:10000] if len(content) > 10000 else content
        # 处置状态验证
        if not isinstance(disposal_status, int) or disposal_status < 0 or disposal_status > 9:
            self.logger.error("disposal_status参数必须为个位正整数")
            return False
                
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            return False
        
        if not source_url: 
            source_url = ''
        else:
            source_url = source_url.strip()
        
        # 使用线程锁确保数据库操作的线程安全
        with self.db_handler._db_lock:
            try:
                if 'data' in metadata and metadata['data'] == []:
                    del metadata['data']
                _metadata = json.dumps(metadata, ensure_ascii=False)
                _analytical_notes = json.dumps({
                    'rumor_detection_conclusion': rumor_detection_conclusion,
                    'harmlevel': harmlevel,
                    'reasoning': reasoning,
                    'suggestions': suggestions,
                    'law_ref': law_ref,
                }, ensure_ascii=False)

                fine_dict = {
                    'id': self.get_urlid(source_url) if source_url and len(source_url) > 5 else sha256(content.encode()).hexdigest(),
                    'content': content,
                    'source_url': source_url,
                    'presentation_form': presentation_form,
                    'publish_platform': publish_platform,
                    'publisher_id': publisher_id,
                    'publish_time': publish_time,
                    'ip_region': ip_region,
                    'metadata': _metadata,
                    'labels': labels,
                    'grading': grading,
                    'classification': classification,
                    'is_illegal': is_illegal,
                    'is_aigc': is_aigc,
                    'analytical_notes': _analytical_notes,
                    'disposal_status': disposal_status,
                    'attachment_count': attachment_count,
                    'ref_urls': ref_urls,
                    'create_time': int(time.time()), # 秒级时间戳
                }

                # 构建插入SQL语句
                columns = list(fine_dict.keys())
                placeholders = ['?' for _ in columns]
                values = list(fine_dict.values())
                
                # 使用DuckDB的INSERT OR REPLACE语法
                insert_sql = f"""
                    INSERT OR REPLACE INTO detection_history 
                    ({','.join(columns)})
                    VALUES ({','.join(placeholders)})
                """
                
                self.db_handler.conn.execute(insert_sql, values)
                self.db_handler.conn.commit()
                self.logger.debug("检测历史记录已保存")
                return True
                
            except Exception as e:
                self.logger.error(f"数据库操作失败: {e}")
                if self.db_handler.conn:
                    self.db_handler.conn.rollback()
                return False
    
    def update_analytical_notes(self, id: str, data: dict):
        """
        更新检测历史记录的分析笔记
        :param id: 检测历史记录ID
        :param data: 分析笔记字段和值的字典
        :return: 更新结果，成功时返回true，失败时返回false
        """
        # 参数验证
        id = id.strip()
        if not id or not isinstance(id, str):
            self.logger.error(f"无效的id参数: {id}")
            return False
        if not data:
            self.logger.error("data不能为空")
            return False
        
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")
        
        history = self.get_detection_history_with_id(id)
        if not history:
            self.logger.error(f"ID为{id}的检测历史记录不存在")
            return False
        
        analytical_notes = json.loads(history['analytical_notes'])
        analytical_notes.update(data)
        analytical_notes = json.dumps(analytical_notes, ensure_ascii=False)
        
        # 使用线程锁确保数据库操作的线程安全
        with self.db_handler._db_lock:
            try:
                self.db_handler.conn.execute("""
                    UPDATE detection_history
                    SET analytical_notes = ?
                    WHERE id = ?
                """, (analytical_notes, id))
                self.db_handler.conn.commit()
                self.logger.debug(f"更新分析笔记成功: id={id}")
                return True
            except Exception as e:
                self.logger.error(f"更新分析笔记失败: {e}")
                if self.db_handler.conn:
                    self.db_handler.conn.rollback()
                return False
    
    def update_disposal_status(self, id: str, disposal_status: int):
        """
        更新检测记录的处置状态

        Args:
            id: 检测记录的ID
            disposal_status: 处置状态
        """
        # 处置状态验证
        if not isinstance(disposal_status, int) or disposal_status < 0 or disposal_status > 9:
            raise ValueError("disposal_status参数必须为个位正整数")
        id = id.strip()
        if not id or not isinstance(id, str):
            self.logger.error(f"无效的id参数: {id}")
            raise ValueError("id参数不能为空且必须为字符串")
        
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")

        # 使用线程锁确保数据库操作的线程安全
        with self.db_handler._db_lock:
            try:
                self.db_handler.conn.execute("""
                    UPDATE detection_history
                    SET disposal_status = ?
                    WHERE id = ?
                """, (disposal_status, id))
                self.db_handler.conn.commit()
                self.logger.debug(f"更新处置状态成功: id={id}, disposal_status={disposal_status}")
            except Exception as e:
                self.logger.error(f"更新处置状态失败: {e}")
                if self.db_handler.conn:
                    self.db_handler.conn.rollback()
                raise Exception(f"数据库操作失败: {str(e)}")
    
    def get_detection_history_with_id(self, id: str, flatten_nested: bool = False) -> Dict[str, Any]:
        """
        根据id获取检测历史记录
        
        Args:
            id: 检测记录的ID
            flatten_nested: 是否将嵌套字段展平，默认False
            
        Returns:
            Dict[str, Any]: 历史记录字典
        """
        id = id.strip()
        if not id or not isinstance(id, str):
            self.logger.error(f"无效的id参数: {id}")
            raise ValueError("id参数不能为空且必须为字符串")
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")
        # 使用线程锁确保数据库操作的线程安全
        with self.db_handler._db_lock:
            try:
                result = self.db_handler.conn.execute("""
                    SELECT * FROM detection_history 
                    WHERE id = ?
                """, (id,)).fetchone()
                
                if result:
                    # 获取列名
                    columns = [desc[0] for desc in self.db_handler.conn.description]
                    record_dict = dict(zip(columns, result))

                    # 对指定字段进行JSON解析尝试
                    for field in self.json_fields:
                        if field in record_dict and record_dict[field] is not None:
                            try:
                                # 尝试解析JSON字符串
                                if flatten_nested:
                                    record_dict.update(json.loads(record_dict[field]))
                                    # 删除原始字段，避免重复
                                    del record_dict[field]
                                else:
                                    record_dict[field] = json.loads(record_dict[field])
                            except json.JSONDecodeError:
                                # 如果解析失败，保留原始字符串并记录警告
                                self.logger.warning(
                                    f"字段 {field} 的值不是有效的JSON格式，保持原始值"
                                )
                    
                    return record_dict
                else:
                    self.logger.debug(f"未找到id为 {id} 的记录")
                    return {}
            except Exception as e:
                self.logger.error(f"查询id为 {id} 的记录失败: {e}")
                raise Exception(f"数据库查询失败: {str(e)}")

    def get_detection_history_with_content_and_url(self, content: str, source_url: str = '', flatten_nested: bool = False) -> Dict[str, Any]:
        """
        根据内容和url获取检测历史记录
        Args:
            content: 检测内容
            source_url: 检测内容来源url，可选
            flatten_nested: 是否将嵌套字段展平，默认False
        Returns:
            Dict[str, Any]: 历史记录字典
        """
        content = content.strip()
        if not content:
            self.logger.warning("content参数不能为空")
            return {}
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")
        if not source_url: 
            source_url = ''
        else:
            source_url = source_url.strip()
        
        # 截断过长的内容以防止数据库错误
        content = content[:10000] if len(content) > 10000 else content
        id = self.get_urlid(source_url) if source_url and len(source_url) > 5 else sha256(content.encode()).hexdigest()
        result = self.get_detection_history_with_id(id, flatten_nested=flatten_nested)
        if not result:
            return {}
        return result
    
    def get_detection_history_with_url(self, source_url: str, flatten_nested: bool = False) -> Dict[str, Any]:
        """
        根据url获取检测历史记录
        Args:
            source_url: 检测内容来源url，可选
            flatten_nested: 是否将嵌套字段展平，默认False
        Returns:
            Dict[str, Any]: 历史记录字典
        """
        source_url = source_url.strip()
        if not source_url or len(source_url) < 6:
            self.logger.warning("source_url参数不能为空")
            return {}
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")
        
        id = self.get_urlid(source_url)
        result = self.get_detection_history_with_id(id, flatten_nested=flatten_nested)
        if not result:
            return {}
        return result

    def get_detection_history_with_disposal_status(self, disposal_status: int, flatten_nested: bool = False) -> List[Dict[str, Any]]:
        """
        根据处置状态获取检测历史记录
        Args:
            disposal_status: 处置状态
            flatten_nested: 是否将嵌套字段展平，默认False
        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        # 处置状态验证
        if not isinstance(disposal_status, int) or disposal_status < 0 or disposal_status > 9:
            raise ValueError("disposal_status参数必须为个位正整数")
        if not self.db_handler.conn:
            self.logger.error("数据库连接未建立")
            raise Exception("数据库连接异常")
        # 使用线程锁确保数据库操作的线程安全
        with self.db_handler._db_lock:
            try:
                results = self.db_handler.conn.execute("""
                    SELECT * FROM detection_history 
                    WHERE disposal_status = ?
                    ORDER BY create_time DESC
                """, (disposal_status,)).fetchall()
                
                if results:
                    # 获取列名
                    columns = [desc[0] for desc in self.db_handler.conn.description]
                    records = []
                    
                    for result in results:
                        record_dict = dict(zip(columns, result))
                        
                        # 对指定字段进行JSON解析尝试
                        for field in self.json_fields:
                            if field in record_dict and record_dict[field] is not None:
                                try:
                                    # 尝试解析JSON字符串
                                    if flatten_nested:
                                        record_dict.update(json.loads(record_dict[field]))
                                        # 删除原始字段，避免重复
                                        del record_dict[field]
                                    else:
                                        record_dict[field] = json.loads(record_dict[field])
                                except json.JSONDecodeError:
                                    # 如果解析失败，保留原始字符串并记录警告
                                    self.logger.warning(
                                        f"字段 {field} 的值不是有效的JSON格式，保持原始值"
                                    )
                        records.append(record_dict)
                    
                    self.logger.debug(f"获取到 {len(records)} 条处置状态为 {disposal_status} 的历史记录")
                    return records
                else:
                    self.logger.debug(f"未找到处置状态为 {disposal_status} 的记录")
                    return []
            except Exception as e:
                self.logger.error(f"查询处置状态为 {disposal_status} 的记录失败: {e}")
                raise Exception(f"数据库查询失败: {str(e)}")
