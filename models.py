from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('postgresql://mivankiv@localhost:5432/ai_spring', echo=True)

Base = declarative_base()


class Customer(Base):
    __tablename__ = 'customers'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    face_img_path = Column(String)
    embedding_img_path = Column(String)
    description = Column(String)
    preferences = Column(String)

    def __init__(self, name, email, phone, face_img_path, embedding_img_path, description="Loved customer", preferences="Beer; Wine; Food"):
        self.name = name
        self.email = email
        self.phone = phone
        self.face_img_path = face_img_path
        self.embedding_img_path = embedding_img_path
        self.description = description
        self.preferences = preferences

    def __repr__(self):
        return "<User('%s','%s', '%s')>" % (self.name, self.email, self.phone)




metadata = Base.metadata
metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

if __name__ == "__main__":
    pass
    # testCustomer = Customer("Name", "Email", "38064131211", "1.png", "1.npy")
    # session.add(testCustomer)
    # session.commit()
    # ourUser = session.query(Customer).first()
    # print(ourUser)


