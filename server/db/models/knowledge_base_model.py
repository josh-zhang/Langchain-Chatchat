from sqlalchemy import Column, Integer, String, DateTime, func, Boolean

from server.db.base import Base


class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    __tablename__ = 'knowledge_base'
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    kb_name = Column(String(50), comment='知识库名称')
    kb_info = Column(String(200), comment='知识库简介(用于Agent)')
    kb_agent_guide = Column(String(200), comment='知识库Agent介绍')
    kb_summary = Column(String(500), comment='知识库总结')
    vs_type = Column(String(50), comment='向量库类型')
    search_enhance = Column(Boolean, comment='是否进行检索增强')
    embed_model = Column(String(50), comment='嵌入模型名称')
    file_count = Column(Integer, default=0, comment='文件数量')
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    def __repr__(self):
        return (
            f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_info='{self.kb_info},kb_agent_guide='{self.kb_agent_guide}, "
            f"kb_summary='{self.kb_summary}, vs_type='{self.vs_type}', embed_model='{self.embed_model}', "
            f"file_count='{self.file_count}', create_time='{self.create_time}')>")
