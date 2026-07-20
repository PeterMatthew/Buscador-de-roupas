import fs from "node:fs";

const ip = process.env.DROPLET_IP;
const content = `/backend/* http://${ip}/:splat 200!\n`;

if (!fs.existsSync("public")) {
  fs.mkdirSync("public", { recursive: true });
}

fs.writeFileSync("public/_redirects", content);
console.log(`Generated public/_redirects pointing to http://${ip}`);
