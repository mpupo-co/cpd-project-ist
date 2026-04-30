# CPD-project

The project consists in the implementation of a sequential and two parallel implementations of a Document Classification algorithm. The purpose is to give hands-on experience in parallel programming on both shared- and distributed-memory systems, using OpenMP and MPI, respectively.

Given a set of D documents, with each document classified according to S subjects, and a number C of cabinets, assign documents to cabinets according to the similarity of subjects.
The classification of a document is obtained by giving it a score to each subject, thus obtaining for each document a vector with S dimensions.
The decision of which document goes into which cabinet is made by trying to minimize the overall “distances” of the subjects of the document assigned to a cabinet (maximize subject similarity).
