"""Download selected public release checkpoint; at most two network attempts."""
import hashlib,json,subprocess
from common import ROOT

def main():
    releases=json.loads((ROOT/'outputs/physics_repair/repo_releases.json').read_text())
    asset=next(a for r in releases for a in r['assets'] if a['name']=='e13i_0291000.tar.gz')
    dst=ROOT/'data/downloads/nju'/asset['name'];dst.parent.mkdir(parents=True,exist_ok=True)
    expected=asset['digest'].split(':')[1];out=ROOT/'outputs/nju_validation';out.mkdir(parents=True,exist_ok=True)
    status={'url':asset['browser_download_url'],'destination':str(dst),'expected_sha256':expected,'ok':False,'attempts':0}
    for attempt in range(2):
        if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()==expected:status['ok']=True;break
        status['attempts']=attempt+1;partial=dst.with_suffix('.partial');print('Download attempt',attempt+1,flush=True)
        p=subprocess.run(['curl','--fail','--location','--connect-timeout','20','--max-time','180','--retry','0','--output',str(partial),asset['browser_download_url']],capture_output=True,text=True)
        if p.returncode==0:
            digest=hashlib.sha256(partial.read_bytes()).hexdigest();print('SHA256',digest,flush=True)
            if digest==expected:partial.replace(dst);status['ok']=True;break
            status['last_error']='SHA256 mismatch'
        else:status['last_error']=p.stderr[-1000:];print(status['last_error'],flush=True)
    (out/'download_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status,indent=2))

if __name__=='__main__':main()
