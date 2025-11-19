-- 违法有害信息检测历史记录表
CREATE TABLE IF NOT EXISTS detection_history (
    id TEXT PRIMARY KEY,                                  -- 被检测的内容与url合并后的SHA256哈希值作为主键
    content TEXT NOT NULL,                                -- 被检测的原始内容的文本
    source_url TEXT,                                      -- 被检测的原始内容的url
    presentation_form TEXT DEFAULT '文本',                 -- 被检测的原始内容的呈现形式，文本/图文/视频
    publish_platform TEXT,                                -- 被检测的原始内容的发布平台，例如：微博、抖音等
    publisher_id TEXT,                                    -- 被检测的原始内容的发布者ID，例如：微博用户ID、抖音用户ID等
    publish_time INTEGER DEFAULT 0,                       -- 被检测的原始内容的发布时间戳
    ip_region TEXT,                                       -- 被检测的原始内容的发布IP所属区域
    metadata TEXT,                                        -- 被检测的原始内容的其他元数据json字符串
    labels TEXT,                                          -- 检测结果标签，英文逗号分隔，例如：'涉政,谣言'
    grading TEXT,                                         -- 检测结果分级，一级/二级/三级
    classification TEXT,                                  -- 检测结果归类，参照“三级十九类”
    is_illegal BOOLEAN DEFAULT 0,                         -- 被检测的原始内容是否为违法信息
    is_aigc BOOLEAN DEFAULT 0,                            -- 被检测的原始内容是否为AI生成内容
    analytical_notes TEXT,                                -- 检测结果分析备注，json字符串
    disposal_status INTEGER NOT NULL DEFAULT 0,           -- 处置状态，0无须处置，1未推送处置，2已推送处置，3已处置完毕
    attachment_count INTEGER NOT NULL DEFAULT 0,          -- 附件的数量
    ref_urls TEXT,                                        -- 参考信息的链接列表，英文逗号分隔
    create_time INTEGER NOT NULL DEFAULT 0                -- 记录时间戳
);

-- 创建索引以提高查询性能
CREATE INDEX IF NOT EXISTS idx_id ON detection_history(id);
CREATE INDEX IF NOT EXISTS idx_source_url ON detection_history(source_url);
CREATE INDEX IF NOT EXISTS idx_presentation_form ON detection_history(presentation_form);
CREATE INDEX IF NOT EXISTS idx_publish_platform ON detection_history(publish_platform);
CREATE INDEX IF NOT EXISTS idx_publisher_id ON detection_history(publisher_id);
CREATE INDEX IF NOT EXISTS idx_publish_time ON detection_history(publish_time);
CREATE INDEX IF NOT EXISTS idx_ip_region ON detection_history(ip_region);
CREATE INDEX IF NOT EXISTS idx_labels ON detection_history(labels);
CREATE INDEX IF NOT EXISTS idx_grading ON detection_history(grading);
CREATE INDEX IF NOT EXISTS idx_classification ON detection_history(classification);
CREATE INDEX IF NOT EXISTS idx_is_illegal ON detection_history(is_illegal);
CREATE INDEX IF NOT EXISTS idx_is_aigc ON detection_history(is_aigc);
CREATE INDEX IF NOT EXISTS idx_disposal_status ON detection_history(disposal_status);
