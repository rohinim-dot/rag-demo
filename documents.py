"""
Knowledge base for the RAG demo.

This is a small set of "documents" (short text chunks) that the system
will search over. In a real RAG system this might be thousands of
pages from a company wiki, product docs, or support tickets — split
into chunks and stored in a vector database.

I used AWS/cloud concepts here since it's a topic I actually know,
which makes it easier to sanity-check whether the retrieved answers
are actually correct.
"""

DOCUMENTS = [
    {
        "id": "doc1",
        "text": "Amazon EC2 (Elastic Compute Cloud) provides resizable virtual "
                "servers in the cloud, called instances. You choose instance "
                "types based on CPU, memory, and networking needs, and can "
                "scale the number of instances up or down based on demand.",
    },
    {
        "id": "doc2",
        "text": "Amazon S3 (Simple Storage Service) is an object storage "
                "service used to store and retrieve any amount of data, such "
                "as files, backups, and static website assets. Data is stored "
                "in containers called buckets.",
    },
    {
        "id": "doc3",
        "text": "AWS IAM (Identity and Access Management) lets you control who "
                "can access AWS resources and what actions they can perform. "
                "IAM policies define permissions, and following the principle "
                "of least privilege means granting only the access someone "
                "actually needs.",
    },
    {
        "id": "doc4",
        "text": "Amazon VPC (Virtual Private Cloud) lets you provision a "
                "logically isolated section of AWS where you can launch "
                "resources in a virtual network you define, including subnets, "
                "route tables, and gateways.",
    },
    {
        "id": "doc5",
        "text": "AWS Lambda is a serverless compute service that runs your "
                "code in response to events, such as file uploads or API "
                "calls, without you needing to provision or manage servers. "
                "You are billed only for the compute time you consume.",
    },
    {
        "id": "doc6",
        "text": "Terraform is an Infrastructure as Code tool that lets you "
                "define cloud resources in configuration files, so "
                "infrastructure can be version-controlled, reviewed, and "
                "reliably reproduced across environments.",
    },
    {
        "id": "doc7",
        "text": "Amazon CloudWatch collects logs, metrics, and events from AWS "
                "resources so you can monitor performance, set alarms, and "
                "troubleshoot issues in near real-time.",
    },
    {
        "id": "doc8",
        "text": "AWS Security Groups act as virtual firewalls for EC2 "
                "instances, controlling inbound and outbound traffic at the "
                "instance level based on rules you define, such as allowed "
                "ports and IP ranges.",
    },
]
