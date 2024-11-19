import requests
import sys

def get_all_starred_repos(username):
    all_repos = set()
    page = 1
    
    while True:
        url = f"https://api.github.com/users/{username}/starred?page={page}&per_page=100"
        response = requests.get(url)
        
        if response.status_code != 200:
            break
            
        repos = response.json()
        if not repos:
            break
            
        for repo in repos:
            all_repos.add(repo['html_url'])
        
        page += 1
    
    return sorted(all_repos)

# Get all starred repos
repos = get_all_starred_repos('sosacrazy126')

# Write to file
with open('all_starred_repos.txt', 'w') as f:
    for repo in repos:
        f.write(repo + '\n')

print(f"Found {len(repos)} unique starred repositories")
